import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
import ptwt
from torch.amp import autocast
import torch.nn.functional as F
import torch.nn.parallel


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, filt_size=5, learnable=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BlurPool(nn.Module):
    def __init__(self, channels, pad_type="reflect", filt_size=4, stride=2, pad_off=0, learnable=False):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        self.learnable = bool(learnable)

        if self.filt_size == 1:
            a = np.array(
                [
                    1.0,
                ]
            )
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

        if self.learnable:
            base = torch.from_numpy(a)
            self.w = nn.Parameter(torch.log(torch.expm1(base + 1e-3)))  # shape [k]
        else:
            k2 = torch.from_numpy(a[:, None] * a[None, :])
            k2 = (k2 / k2.sum()).view(1, 1, self.filt_size, self.filt_size)
            self.register_buffer("filt", k2.repeat(self.channels, 1, 1, 1))

    def _make_kernel(self, x):
        a = F.softplus(self.w)  # ≥0, shape [k]
        a = a / (a.sum() + 1e-12)
        k2 = torch.outer(a, a)  # [k,k]
        k2 = k2 / (k2.sum() + 1e-12)
        return k2[None, None].to(dtype=x.dtype, device=x.device).repeat(x.size(1), 1, 1, 1)  # [channels, 1,k,k]

    def get_kernel(self):
        a = F.softplus(self.w)
        return a / (a.sum() + 1e-12)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            else:
                return self.pad(inp)[:, :, :: self.stride, :: self.stride]

        if self.learnable:
            filt = self._make_kernel(inp)
        else:
            filt = self.filt
        filt = filt.to(dtype=inp.dtype, device=inp.device)
        return F.conv2d(self.pad(inp), filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class LipReg_aa(nn.Module):
    def __init__(
        self, in_chans=3, num_classes=1000, 
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
        layer_scale_init_value=1e-6, head_init_scale=1.,
        wavelet_level=1,
        wavelet_method="haar",
        jacobian_delta=0.5,
        k=1,
        filter_size=5,
        learnable=True,
    ):
        super().__init__()
        self.wavelet_level = wavelet_level
        self.wavelet_method = wavelet_method
        self.num_classes = num_classes
        self.__penalty = 0
        self.jacobian_delta = jacobian_delta
        self.k = k
        
        
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=1),
            BlurPool(dims[0], stride=4, filt_size=filter_size, learnable=learnable),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=1),
                    BlurPool(dims[i+1], stride=2, filt_size=filter_size, learnable=learnable),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def __decompose_high_low_domains(self, batch: torch.Tensor):
        with autocast(enabled=False, device_type="cuda"):
            batch_fp32 = batch.float()

            coeffs = ptwt.wavedec2(
                batch_fp32,
                self.wavelet_method,
                level=self.wavelet_level,
                mode="constant",
            )

            high_freq = self.__reconstruct_from_coeffs(
                coeffs, keep_approx=False, keep_details=True, wavelet_method=self.wavelet_method,
            )
            low_freq = self.__reconstruct_from_coeffs(
                coeffs, keep_approx=True, keep_details=False, wavelet_method=self.wavelet_method,
            )

        return high_freq, low_freq


    @staticmethod
    def __reconstruct_from_coeffs(
        coeffs: list, keep_approx: bool = True, keep_details: bool = True, wavelet_method: str = "haar"
    ) -> torch.Tensor:
        modified_coeffs = []
        for i, level in enumerate(coeffs):
            if i == 0:  # Approximation coefficients
                if keep_approx:
                    modified_coeffs.append(level.float())
                else:
                    modified_coeffs.append(torch.zeros_like(level, dtype=torch.float32))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(tuple(d.float() for d in level))
                else:
                    modified_coeffs.append(
                        tuple(torch.zeros_like(d, dtype=torch.float32) for d in level)
                    )
        with autocast(enabled=False, device_type="cuda"):
            out = ptwt.waverec2(modified_coeffs, wavelet_method)
        return out

    def __jacobian_penalty(self, x, sigma=0.25):
        bs = x.shape[0]
        w = x.shape[-1]
        k = (w / 32)**2
        with autocast(enabled=False, device_type="cuda"):
            x_fp32 = x.float()
            noise = torch.randn_like(x_fp32) * sigma
            out1 = self.forward_features(x_fp32)
            out2 = self.forward_features(x_fp32 + noise)
            difference = out1 - out2
            norm = torch.norm(difference, p="fro")
        # penalty = self.jacobian_delta * norm * norm/ (bs)
        penalty = self.jacobian_delta * norm / (bs*k)
        return penalty

    def pre_penalty(self, *args, **kwargs):
        return self.__penalty

    def post_penalty(self, x, *args, **kwargs) -> None:
        self.clamp_grad(x)

    @property
    def penalty(self):
        return self.pre_penalty()

    def clamp_grad(self, x):
        # print("GRad", x.grad)
        g_high = self.x_high.grad
        mask = g_high.abs() > self.k * x.grad.abs()
        x.grad[mask] = torch.clamp(x.grad[mask], min=-self.k, max=self.k)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        self.x_high, x_low = self.__decompose_high_low_domains(x)
        self.x_high.requires_grad_(True)
        self.x_high.retain_grad()
        x_decompose = self.x_high + x_low
        if self.training:
            self.__penalty = self.__jacobian_penalty(x_decompose)
        x = self.forward_features(x_decompose)
        x = self.head(x)
        return x


from timm.models.registry import register_model

@register_model
def lipreg_aa(pretrained=False, **kwargs):
    model = LipReg_aa(depths=[3,3,27,3], dims=[128,256,512,1024], **kwargs)
    return model