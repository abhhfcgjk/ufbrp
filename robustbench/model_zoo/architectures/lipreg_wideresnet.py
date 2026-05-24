from typing import Tuple

import ptwt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    @property
    def penalty(self):
        return 0


class BasicBlockBlur(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0,
                 filt_size=4, learnable=False):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)

        if stride > 1:
            self.blur1 = BlurPool(
                channels=in_planes,
                filt_size=filt_size,
                stride=stride,
                learnable=learnable,
            )
            conv1_stride = 1
        else:
            self.blur1 = nn.Identity()
            conv1_stride = stride

        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=conv1_stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)

        if not self.equalInOut:
            if stride > 1:
                self.shortcut_blur = BlurPool(
                    channels=in_planes,
                    filt_size=filt_size,
                    stride=stride,
                    learnable=learnable,
                )
                shortcut_stride = 1
            else:
                self.shortcut_blur = nn.Identity()
                shortcut_stride = stride
            self.convShortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=shortcut_stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut_blur = nn.Identity()
            self.convShortcut = None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out_in = x
        else:
            out_in = self.relu1(self.bn1(x))
        out = self.blur1(out_in)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(self.shortcut_blur(x))
        return torch.add(shortcut, out)

class NetworkBlockBlur(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, filter_size=3, learnable=False):
        super(NetworkBlockBlur, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, filter_size, learnable)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, filter_size, learnable):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, filter_size, learnable))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet_Blur(nn.Module):
    """ Based on code from https://github.com/yaodongyu/TRADES """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True, filter_size=3, learnable=False):
        super(WideResNet_Blur, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlockBlur
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlockBlur(n, nChannels[0], nChannels[1], block, 1, dropRate, filter_size, learnable)
        if sub_block1:
            # 1st sub-block
            self.sub_block1 = NetworkBlockBlur(n, nChannels[0], nChannels[1], block, 1, dropRate, filter_size, learnable)
        # 2nd block
        self.block2 = NetworkBlockBlur(n, nChannels[1], nChannels[2], block, 2, dropRate, filter_size, learnable)
        # 3rd block
        self.block3 = NetworkBlockBlur(n, nChannels[2], nChannels[3], block, 2, dropRate, filter_size, learnable)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes, bias=bias_last)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not m.bias is None:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def WideResNetBlur_L(filter_size, learnable, num_classes=10):
    return WideResNet_Blur(num_classes=num_classes, depth=94, widen_factor=16, filter_size=filter_size, learnable=learnable)


def WideResNet_L( num_classes=10):
    return WideResNet(num_classes=num_classes, depth=94, widen_factor=16)

class LipReg_aa_WideResNet(nn.Module):
    def __init__(
        self,
        wavelet_level=2,
        wavelet_method="haar",
        num_classes=10,
        jacobian_delta=0.5,
        k=1.0,
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
        self.filter_size = filter_size
        self.learnable = learnable

        model = WideResNetBlur_L(num_classes=self.num_classes, filter_size=self.filter_size, learnable=self.learnable)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=1024, out_features=self.num_classes, bias=True)

    def __decompose_high_low_domains(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decomposes a batch of images into high and low frequency domains using wavelet transform.

        Args:
            batch: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (high_freq, low_freq) tensors with same shape as input
        """
        coeffs = ptwt.wavedec2(batch, self.wavelet_method, level=self.wavelet_level, mode="constant")
        high_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=False, keep_details=True, wavelet_method=self.wavelet_method
        )
        low_freq = self.__reconstruct_from_coeffs(
            coeffs, keep_approx=True, keep_details=False, wavelet_method=self.wavelet_method
        )
        return high_freq, low_freq

    @staticmethod
    def __reconstruct_from_coeffs(
        coeffs: list, keep_approx: bool = True, keep_details: bool = True, wavelet_method: str = "haar"
    ) -> torch.Tensor:
        """
        Reconstructs image from modified wavelet coefficients.

        Args:
            coeffs: Wavelet coefficients from ptwt.wavedec2
            keep_approx: Whether to keep approximation coefficients
            keep_details: Whether to keep detail coefficients

        Returns:
            Reconstructed tensor
        """
        modified_coeffs = []
        for i, level in enumerate(coeffs):
            if i == 0:  # Approximation coefficients
                modified_coeffs.append(level if keep_approx else torch.zeros_like(level))
            else:  # Detail coefficients
                if keep_details:
                    modified_coeffs.append(level)
                else:
                    modified_coeffs.append(tuple(torch.zeros_like(d) for d in level))
        return ptwt.waverec2(modified_coeffs, wavelet_method)

    def __jacobian_penalty(self, x, sigma=0.25):
        bs = x.shape[0]
        noise = torch.randn_like(x) * sigma
        difference = self.model(x) - self.model(x + noise)
        norm = torch.norm(difference, p="fro")
        return self.jacobian_delta * norm * norm / bs

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

    def forward(self, x: Tensor) -> Tensor:
        self.x_high, x_low = self.__decompose_high_low_domains(x)
        self.x_high.requires_grad_(True)
        self.x_high.retain_grad()
        x_low = x_low.detach()
        x_decompose = self.x_high + x_low
        self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output
    


from collections import OrderedDict


class ImageNormalizer(nn.Module):
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        persistent: bool = True,
    ):
        super(ImageNormalizer, self).__init__()

        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=persistent)
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=persistent)

    def forward(self, inputs: torch.Tensor):
        return (inputs - self.mean) / self.std


def create_model(
    model: nn.Module,
    mean: tuple[float, float, float]=(0.4914, 0.4822, 0.4465),
    std: tuple[float, float, float]=(0.2023, 0.1994, 0.2010),
):
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)

# def WideResNet_LipReg(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
#     model = LipReg_aa_WideResNet()
#     return create_model(model, mean, std)