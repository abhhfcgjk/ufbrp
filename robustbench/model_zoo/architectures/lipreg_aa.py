from typing import Tuple

import ptwt
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.amp import autocast


"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""



# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import numpy as np


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
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, filter_size=3, learnable=False):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(
                BlurPool(filt_size=filter_size, channels=in_planes, stride=stride, learnable=learnable),
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # self.shortcut = nn.Sequential(
            # nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            # nn.BatchNorm2d(self.expansion * planes),
            # )
            self.shortcut = nn.Sequential(
                *(
                    (
                        [BlurPool(filt_size=filter_size, stride=stride, channels=in_planes, learnable=learnable)]
                        if stride != 1
                        else []
                    )
                    + [
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.expansion * planes),
                    ]
                )
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, filter_size=3, learnable=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        else:
            self.conv3 = nn.Sequential(
                BlurPool(planes, filt_size=filter_size, stride=stride, learnable=learnable),
                nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                *(
                    (
                        [BlurPool(filt_size=filter_size, stride=stride, channels=in_planes, learnable=learnable)]
                        if stride != 1
                        else []
                    )
                    + [
                        nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=1, bias=False),
                        nn.BatchNorm2d(self.expansion * planes),
                    ]
                )
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, filter_size=3, learnable=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.is_imagenet = num_classes == 1000
        if self.is_imagenet:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=1),
                BlurPool(filt_size=filter_size, stride=2, channels=64, learnable=learnable),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            block, 128, num_blocks[1], stride=2, filter_size=filter_size, learnable=learnable
        )
        self.layer3 = self._make_layer(
            block, 256, num_blocks[2], stride=2, filter_size=filter_size, learnable=learnable
        )
        self.layer4 = self._make_layer(
            block, 512, num_blocks[3], stride=2, filter_size=filter_size, learnable=learnable
        )

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, filter_size=1, learnable=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, filter_size, learnable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.is_imagenet:
            out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @property
    def penalty(self):
        return 0


def ResNet18(filter_size=3, num_classes=10, learnable=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, filter_size=filter_size, learnable=learnable)


def ResNet34Blur(filter_size=3, num_classes=10, learnable=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, filter_size=filter_size, learnable=learnable)


def ResNet50Blur(filter_size=3, num_classes=10, learnable=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, filter_size=filter_size, learnable=learnable)


class LipReg_aa(nn.Module):
    def __init__(
        self,
        wavelet_level=4,
        wavelet_method="haar",
        num_classes=10,
        jacobian_delta=0.5,
        k=0.5,
        filter_size=5,
        learnable=False,
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

        model = ResNet50Blur(num_classes=self.num_classes, filter_size=self.filter_size, learnable=self.learnable)
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes, bias=True)

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
            out1 = self.model(x_fp32)
            out2 = self.model(x_fp32 + noise)
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

    def forward(self, x: Tensor) -> Tensor:
        self.x_high, x_low = self.__decompose_high_low_domains(x)
        self.x_high.requires_grad_(True)
        self.x_high.retain_grad()
        # x_low = x_low.detach()
        x_decompose = self.x_high + x_low
        if self.training:
            self.__penalty = self.__jacobian_penalty(x_decompose)
        feature = self.model(x_decompose)
        pool = self.avgpool(feature)
        pool = pool.view(pool.size(0), -1)
        output = self.fc(pool)
        return output


# class LipReg_aa(nn.Module):
#     def __init__(
#         self,
#         wavelet_level=4,
#         wavelet_method="haar",
#         num_classes=10,
#         jacobian_delta=0.5,
#         k=0.5,
#         filter_size=5,
#         learnable=False,
#         sigma=8 / 255,   # noise scale for Jacobian penalty
#     ):
#         super().__init__()
#         self.wavelet_level = wavelet_level
#         self.wavelet_method = wavelet_method
#         self.num_classes = num_classes
#         self.jacobian_delta = jacobian_delta
#         self.k = k
#         self.filter_size = filter_size
#         self.learnable = learnable
#         self.sigma = sigma

#         backbone = ResNet50Blur(
#             num_classes=self.num_classes,
#             filter_size=self.filter_size,
#             learnable=self.learnable,
#         )

#         # backbone without final linear layer
#         self.backbone = nn.Sequential(*list(backbone.children())[:-1])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, self.num_classes, bias=True)

#         self.__penalty = 0.0

#     def __decompose_high_low_domains(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
#         device_type = batch.device.type
#         with autocast(device_type=device_type, enabled=False):
#             batch_fp32 = batch.float()

#             coeffs = ptwt.wavedec2(
#                 batch_fp32,
#                 self.wavelet_method,
#                 level=self.wavelet_level,
#                 mode="constant",
#             )

#             high_freq = self.__reconstruct_from_coeffs(
#                 coeffs,
#                 keep_approx=False,
#                 keep_details=True,
#                 wavelet_method=self.wavelet_method,
#             )
#             low_freq = self.__reconstruct_from_coeffs(
#                 coeffs,
#                 keep_approx=True,
#                 keep_details=False,
#                 wavelet_method=self.wavelet_method,
#             )

#         return high_freq, low_freq

#     @staticmethod
#     def __reconstruct_from_coeffs(
#         coeffs: list,
#         keep_approx: bool = True,
#         keep_details: bool = True,
#         wavelet_method: str = "haar",
#     ) -> Tensor:
#         modified_coeffs = []
#         for i, level in enumerate(coeffs):
#             if i == 0:
#                 if keep_approx:
#                     modified_coeffs.append(level.float())
#                 else:
#                     modified_coeffs.append(torch.zeros_like(level, dtype=torch.float32))
#             else:
#                 if keep_details:
#                     modified_coeffs.append(tuple(d.float() for d in level))
#                 else:
#                     modified_coeffs.append(
#                         tuple(torch.zeros_like(d, dtype=torch.float32) for d in level)
#                     )

#         return ptwt.waverec2(modified_coeffs, wavelet_method)

#     def _clamp_high_grad_hook(self, grad: Tensor) -> Tensor:
#         if self.k is None or self.k <= 0:
#             return grad
#         return grad.clamp(min=-self.k, max=self.k)

#     def _build_input(self, x: Tensor, clamp_high_grad: bool = False) -> Tensor:
#         x_high, x_low = self.__decompose_high_low_domains(x)
#         if clamp_high_grad and self.training:
#             if not x_high.requires_grad:
#                 x_high = x_high.detach().requires_grad_(True)
#             x_high.register_hook(self._clamp_high_grad_hook)

#         x_recompose = x_high + x_low
#         return x_recompose

#     def _logits_from_input(self, x: Tensor) -> Tensor:
#         feat = self.backbone(x)
#         feat = self.avgpool(feat)
#         feat = torch.flatten(feat, 1)
#         logits = self.fc(feat)
#         return logits

#     def __jacobian_penalty(self, x: Tensor) -> Tensor:
#         device_type = x.device.type
#         with autocast(device_type=device_type, enabled=False):
#             x_fp32 = x.float()
#             noise = torch.randn_like(x_fp32) * self.sigma.to(x.device)
#             x_noisy = (x_fp32 + noise).clamp(0.0, 1.0)

#             logits_1 = self._logits_from_input(x_fp32)
#             logits_2 = self._logits_from_input(x_noisy)

#             diff = logits_1 - logits_2
#             penalty = self.jacobian_delta * diff.pow(2).sum(dim=1).mean()
#         # print(penalty)
#         return penalty

#     def forward(self, x: Tensor) -> Tensor:
#         x_recompose = self._build_input(x, clamp_high_grad=self.training)
#         logits = self._logits_from_input(x_recompose)

#         if not self.training:
#             self.__penalty = self.__jacobian_penalty(x_recompose)

#         return logits

#     @property
#     def penalty(self):
#         return self.pre_penalty()

#     def pre_penalty(self, *args, **kwargs):
#         return self.__penalty


