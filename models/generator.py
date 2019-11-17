import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from math import sqrt
import random
from model import StyledConvBlock, EqualConv2d
from models.builder import ModuleBuilder


class AlphaMix(nn.Module):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, t1: Tensor, t2: Tensor):
        assert t1.shape[-1] * 2 == t2.shape[-1]
        t1 = F.interpolate(t1, scale_factor=2, mode='nearest')
        return (1 - self.alpha) * t1 + self.alpha * t2


class GeneratorBuilder:

    def __init__(self, fused=True, alpha: float = -1):
        super().__init__()

        self.builder = ModuleBuilder()

        self.builder.add_module("progression_1", StyledConvBlock(512, 512, 3, 1, initial=True),              ["out1", "style1", "noise1"], ["out2"])
        self.builder.add_module("progression_2", StyledConvBlock(512, 512, 3, 1, initial=True),              ["out2", "style2", "noise2"], ["out3"])
        self.builder.add_module("progression_3", StyledConvBlock(512, 512, 3, 1, initial=True),              ["out3", "style3", "noise3"], ["out4"])
        self.builder.add_module("progression_4", StyledConvBlock(512, 512, 3, 1, initial=True),              ["out4", "style4", "noise4"], ["out5"])
        self.builder.add_module("progression_5", StyledConvBlock(512, 256, 3, 1, initial=True),              ["out5", "style5", "noise5"], ["out6"])
        self.builder.add_module("progression_6", StyledConvBlock(256, 128, 3, 1, initial=True, fused=fused), ["out6", "style6", "noise6"], ["out7"])
        self.builder.add_module("progression_7", StyledConvBlock(128, 64, 3, 1, initial=True, fused=fused),  ["out7", "style7", "noise7"], ["out8"])
        self.builder.add_module("progression_8", StyledConvBlock(64, 32, 3, 1, initial=True, fused=fused),   ["out8", "style8", "noise8"], ["out9"])
        self.builder.add_module("progression_9", StyledConvBlock(32, 16, 3, 1, initial=True, fused=fused),   ["out9", "style9", "noise9"], ["out10"])

        self.builder.add_edge(["progression_1"], "progression_2")
        self.builder.add_edge(["progression_2"], "progression_3")
        self.builder.add_edge(["progression_3"], "progression_4")
        self.builder.add_edge(["progression_4"], "progression_5")
        self.builder.add_edge(["progression_5"], "progression_6")
        self.builder.add_edge(["progression_6"], "progression_7")
        self.builder.add_edge(["progression_7"], "progression_8")
        self.builder.add_edge(["progression_8"], "progression_9")

        self.builder.add_module("to_rgb_1", EqualConv2d(512, 3, 1), ["out2"], ["out_rgb1"])
        self.builder.add_module("to_rgb_2", EqualConv2d(512, 3, 1), ["out3"], ["out_rgb2"])
        self.builder.add_module("to_rgb_3", EqualConv2d(512, 3, 1), ["out4"], ["out_rgb3"])
        self.builder.add_module("to_rgb_4", EqualConv2d(512, 3, 1), ["out5"], ["out_rgb4"])
        self.builder.add_module("to_rgb_5", EqualConv2d(256, 3, 1), ["out6"], ["out_rgb5"])
        self.builder.add_module("to_rgb_6", EqualConv2d(128, 3, 1), ["out7"], ["out_rgb6"])
        self.builder.add_module("to_rgb_7", EqualConv2d(64, 3, 1),  ["out8"], ["out_rgb7"])
        self.builder.add_module("to_rgb_8", EqualConv2d(32, 3, 1),  ["out9"], ["out_rgb8"])
        self.builder.add_module("to_rgb_9", EqualConv2d(16, 3, 1),  ["out10"], ["out_rgb9"])

        self.builder.add_edge(["progression_1"], "to_rgb_1")
        self.builder.add_edge(["progression_2"], "to_rgb_2")
        self.builder.add_edge(["progression_3"], "to_rgb_3")
        self.builder.add_edge(["progression_4"], "to_rgb_4")
        self.builder.add_edge(["progression_5"], "to_rgb_5")
        self.builder.add_edge(["progression_6"], "to_rgb_6")
        self.builder.add_edge(["progression_7"], "to_rgb_7")
        self.builder.add_edge(["progression_8"], "to_rgb_8")
        self.builder.add_edge(["progression_9"], "to_rgb_9")

        self.builder.add_module("alpha_mix_1", nn.Identity, ["out_rgb1"], ["out_rgb1"])
        self.builder.add_module("alpha_mix_2", AlphaMix(alpha), ["out_rgb1", "out_rgb2"], ["out_rgb2"])
        self.builder.add_module("alpha_mix_3", AlphaMix(alpha), ["out_rgb2", "out_rgb3"], ["out_rgb3"])
        self.builder.add_module("alpha_mix_4", AlphaMix(alpha), ["out_rgb3", "out_rgb4"], ["out_rgb4"])
        self.builder.add_module("alpha_mix_5", AlphaMix(alpha), ["out_rgb4", "out_rgb5"], ["out_rgb5"])
        self.builder.add_module("alpha_mix_6", AlphaMix(alpha), ["out_rgb5", "out_rgb6"], ["out_rgb6"])
        self.builder.add_module("alpha_mix_7", AlphaMix(alpha), ["out_rgb6", "out_rgb7"], ["out_rgb7"])
        self.builder.add_module("alpha_mix_8", AlphaMix(alpha), ["out_rgb7", "out_rgb8"], ["out_rgb8"])
        self.builder.add_module("alpha_mix_9", AlphaMix(alpha), ["out_rgb8", "out_rgb9"], ["out_rgb9"])

        self.builder.add_edge(["to_rgb_1", "to_rgb_2"], "alpha_mix_2")
        self.builder.add_edge(["to_rgb_2", "to_rgb_3"], "alpha_mix_3")
        self.builder.add_edge(["to_rgb_3", "to_rgb_4"], "alpha_mix_4")
        self.builder.add_edge(["to_rgb_4", "to_rgb_5"], "alpha_mix_5")
        self.builder.add_edge(["to_rgb_5", "to_rgb_6"], "alpha_mix_6")
        self.builder.add_edge(["to_rgb_6", "to_rgb_7"], "alpha_mix_7")
        self.builder.add_edge(["to_rgb_7", "to_rgb_8"], "alpha_mix_8")
        self.builder.add_edge(["to_rgb_8", "to_rgb_9"], "alpha_mix_9")

    def build(self, step):

        return self.builder.build(
            ["progression_" + str(i+1) for i in range(step + 1)],
            "alpha_mix_" + str(step + 1)
        )
