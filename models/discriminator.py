import torch
from torch import nn, Tensor

from model import EqualConv2d, ConvBlock, EqualLinear
from models.builder import ModuleBuilder, Identity, NamedModule


class Out(nn.Module):

    def forward(self, out):
        out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
        mean_std = out_std.mean()
        mean_std = mean_std.expand(out.size(0), 1, 4, 4)
        out = torch.cat((out, mean_std), 1)
        return out


class AlphaMixWithProgression(nn.Module):

    def __init__(self, progression: nn.Module, alpha: float):
        super().__init__()
        self.progr = progression
        self.alpha = alpha

    def forward(self, x1: Tensor, x2: Tensor):
        x1_pr = self.progr(x1)
        return (1 - self.alpha) * x2 + self.alpha * x1_pr


class ToLinear(nn.Module):
    def forward(self, input: Tensor):
        return input.view(input.shape[0], -1)


class DiscriminatorBuilder:

    def make_from_rgb(self, out_channel):
        if self.from_rgb_activate:
            return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

        else:
            return EqualConv2d(3, out_channel, 1)

    def __init__(self, alpha, fused=True, from_rgb_activate=False):

        self.from_rgb_activate = from_rgb_activate

        builder3 = ModuleBuilder()

        builder3.add_module_seq(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("input_%d" % i,
                       Identity(),
                       ["rgb%d" % i],
                       ["rgb%d" % i])
        )

        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("down_pool2x_%d" % i,
                       nn.AvgPool2d(2),
                       ["rgb%d" % i],
                       ["rgb%d" % (i - 1)])
        )

        builder3.add_edge_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ([f"input_{i}"], f"down_pool2x_{i}")
        )

        builder3.add_module_seq(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ("from_rgb_%d" % i,
                       self.make_from_rgb(512 if i < 5 else 512 // (2 ** (i - 4))),
                       ["rgb%d" % i],
                       ["out%d" % i])
        )

        builder3.add_edge_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ([f"down_pool2x_{i}"], f"from_rgb_{i - 1}")
        )

        builder3.add_edge_seq(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ([f"input_{i}"], f"from_rgb_{i}")
        )

        builder3.add_module(
            "progression_1",
            nn.Sequential(Out(), ConvBlock(513, 512, 3, 1, 4, 0), ToLinear(), EqualLinear(512, 1)),
            ["out_pr1"], ["out_pr0"]
        )
        builder3.add_module_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: (f"progression_{i}",
                       ConvBlock(512 if i < 5 else 512 // (2 ** (i - 4)), 512 if i < 6 else 512 // (2 ** (i - 5)), 3, 1,
                                 downsample=True, fused=fused if i >= 6 else False),
                       [f"out_pr{i}"],
                       [f"out_pr{i-1}"])
        )

        builder3.add_module_seq(
            [1, 2, 3, 4, 5, 6, 7, 8],
            lambda i: (f"alpha_mix_pr_{i}",
                       AlphaMixWithProgression(builder3.nodes[f"progression_{i}"].module, alpha=alpha),
                       [f"out{i+1}", f"out{i}"],
                       [f"out_pr{i}"])
        )

        builder3.add_edge_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ([f"from_rgb_{i}", f"from_rgb_{i-1}"], f"alpha_mix_pr_{i-1}")
        )

        builder3.add_edge_seq(
            [1, 2, 3, 4, 5, 6, 7, 8],
            lambda i: ([f"alpha_mix_pr_{i}"], f"progression_{i}")
        )

        builder3.add_edge_seq(
            [2, 3, 4, 5, 6, 7, 8, 9],
            lambda i: ([f"progression_{i}"], f"progression_{i-1}")
        )

        self.builder = builder3

    def build(self, step) -> NamedModule:

        return self.builder.build(
            ["input_" + str(step + 1)],
            "progression_1"
        )









