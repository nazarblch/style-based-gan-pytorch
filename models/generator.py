import torch
from torch import nn, Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from math import sqrt
import random
from model import StyledConvBlock, EqualConv2d
from models.builder import ModuleBuilder, Identity


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

        builder3 = ModuleBuilder()

        builder3.add_module("input_1", Identity(), ["input", "style1", "noise1"], ["out1", "style1", "noise1"])
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("input_%d" % i, Identity(), ["style%d" % i, "noise%d" % i], ["style%d" % i, "noise%d" % i])
        )

        builder3.add_module("progression_1", StyledConvBlock(512, 512, 3, 1, initial=True),
                            ["out1", "style1", "noise1"], ["out2"])
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("progression_%d" % i,
                       StyledConvBlock(512 if i < 6 else 512 // (2 ** (i - 5)), 512 if i < 5 else 512 // (2 ** (i - 4)),
                                       3, 1, upsample=True, fused=fused if i >= 6 else False),
                       ["out%d" % i, "style%d" % i, "noise%d" % i],
                       ["out%d" % (i + 1)])
        )

        builder3.add_edge(["input_1"], "progression_1")
        builder3.add_edge_seq([2, 3, 4, 5, 6, 7, 8, 9], lambda i: ([f"input_{i}", f"progression_{i - 1}"], f"progression_{i}"))

        builder3.add_module_seq(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("to_rgb_%d" % i,
                       EqualConv2d(512 if i < 5 else 512 // (2 ** (i - 4)), 3, 1),
                       ["out%d" % (i + 1)],
                       ["out_rgb%d" % i])
        )

        builder3.add_edge_seq([1, 2, 3, 4, 5, 6, 7, 8, 9],
                              lambda i: ([f"progression_{i}"], f"to_rgb_{i}"))

        builder3.add_module("alpha_mix_1", Identity(), ["out_rgb1"], ["out_rgb1"])
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("alpha_mix_%d" % i,
                       AlphaMix(alpha),
                       ["out_rgb%d" % (i - 1), "out_rgb%d" % i],
                       ["out_rgb%d" % i])
        )

        builder3.add_edge_seq([2, 3, 4, 5, 6, 7, 8, 9], lambda i: ([f"to_rgb_{i - 1}", f"to_rgb_{i}"], f"alpha_mix_{i}"))

        self.builder = builder3

    def build(self, step):

        return self.builder.build(
            ["input_" + str(i+1) for i in range(step + 1)],
            "alpha_mix_" + str(step + 1)
        )
