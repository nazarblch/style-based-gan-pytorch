import torch
import unittest
from torch import nn

from model import Generator, EqualConv2d, StyledConvBlock
from models.builder import ModuleBuilder, Identity
from models.discriminator import DiscriminatorBuilder
from models.generator import AlphaMix, GeneratorBuilder

builder = ModuleBuilder()

builder.add_module("1", nn.Identity(), ["1x"], ["1x"])
builder.add_module("2", nn.Identity(), ["2x"], ["2x"])
builder.add_module("3", nn.Identity(), ["1x", "2x"], ["3x"])
builder.add_module("4", nn.Identity(), ["2x", "3x"], ["4x"])
builder.add_module("5", nn.Identity(), ["4x"], ["5x1", "5x2"])
builder.add_module("6", nn.Identity(), ["3x", "5x1"], ["6x"])
builder.add_module("7", nn.Identity(), ["5x2"], ["7x"])
builder.add_module("8", nn.Identity(), ["6x"], ["8x"])

builder.add_edge(["1", "2"], "3")
builder.add_edge(["2", "3"], "4")
builder.add_edge(["4"], "5")
builder.add_edge(["5"], "7")
builder.add_edge(["3", "5"], "6")
builder.add_edge(["6"], "8")

builder.plot('g_test')


class TwoArgNet(nn.Module):

    def __init__(self, inc, outc):
        super().__init__()
        self.layer = nn.Linear(inc, outc)

    def forward(self, t1, t2):
        return self.layer(torch.cat((t1, t2), dim=1)).sigmoid()


net1 = nn.Linear(10, 10)
net2 = TwoArgNet(25, 12)
net3 = nn.Linear(5, 15)
net4 = TwoArgNet(27, 20)
net5 = nn.Linear(20, 5)
net6 = nn.Linear(20, 2)

builder2 = ModuleBuilder()

builder2.add_module("1", net1, ["1x"], ["1x"])
builder2.add_module("2", net2, ["1x", "3x"], ["2x"])
builder2.add_module("3", net3, ["3x"], ["3x"])
builder2.add_module("4", net4, ["2x", "3x"], ["4x"])
builder2.add_module("5", net5, ["4x"], ["5x"])
builder2.add_module("6", net6, ["4x"], ["6x"])

builder2.add_edge(["1", "3"], "2")
builder2.add_edge(["2", "3"], "4")
builder2.add_edge(["4"], "5")
builder2.add_edge(["4"], "6")


builder3 = ModuleBuilder()

builder3.add_module("input_1", Identity(), ["input", "style1", "noise1"], ["out1", "style1", "noise1"])
builder3.add_module_seq(
    [2, 3, 4, 5],
    lambda i: ("input_%d" % i, Identity(), ["style%d" % i, "noise%d" % i], ["style%d" % i, "noise%d" % i])
)

builder3.add_module("progression_1", StyledConvBlock(512, 512, 3, 1, initial=True),  ["out1", "style1", "noise1"], ["out2"])
builder3.add_module_seq(
    [2, 3, 4, 5],
    lambda i: ("progression_%d" % i,
               StyledConvBlock(512 if i < 6 else 512 // (2**(i-5)), 512 if i < 5 else 512 // (2**(i-4)), 3, 1, upsample=True),
               ["out%d" % i, "style%d" % i, "noise%d" % i],
               ["out%d" % (i+1)])
)


builder3.add_edge(["input_1"], "progression_1")
builder3.add_edge_seq([2, 3, 4, 5], lambda i: ([f"input_{i}", f"progression_{i-1}"], f"progression_{i}"))

builder3.add_module_seq(
    [1, 2, 3, 4, 5],
    lambda i: ("to_rgb_%d" % i,
               EqualConv2d(512 if i < 5 else 512 // (2**(i-4)), 3, 1),
               ["out%d" % (i+1)],
               ["out_rgb%d" % i])
)


builder3.add_edge_seq([1, 2, 3, 4, 5], lambda i: ([f"progression_{i}"], f"to_rgb_{i}"))

alpha = 0.4
builder3.add_module("alpha_mix_1", Identity(), ["out_rgb1"], ["out_rgb1"])
builder3.add_module_seq(
    [2, 3, 4, 5],
    lambda i: ("alpha_mix_%d" % i,
               AlphaMix(alpha),
               ["out_rgb%d" % (i-1), "out_rgb%d" % i],
               ["out_rgb%d" % i])
)

builder3.add_edge_seq([2, 3, 4, 5], lambda i: ([f"to_rgb_{i-1}", f"to_rgb_{i}"], f"alpha_mix_{i}"))

builder3.plot("gen")


class TestBuilder(unittest.TestCase):

    def test_subgraph(self):

        sub_graph = builder.sub_graph(["1", "2"], "7")
        self.assertSetEqual(set(sub_graph.nodes.keys()), set(["1", "2", "3", "4", "5", "7"]))
        sub_graph = builder.sub_graph(["3", "4"], "8")
        self.assertSetEqual(set(sub_graph.nodes.keys()), set(["3", "4", "5", "6", "8"]))

    def test_module(self):

        module = builder2.build(["1", "3"], "5")

        self.assertListEqual(sorted(module.from_names), ['1x', '3x'])
        self.assertListEqual(module.to_names, ['5x'])

        input = {
            "1x": torch.randn(2, 10),
            "3x": torch.randn(2, 5)
        }

        res = module.forward(input)["5x"]

        res3 = net3(input["3x"])
        res2 = net2(net1(input["1x"]), res3)
        res4 = net4(res2, res3)
        res5 = net5(res4)

        self.assertTrue((res - res5).abs().max().item() < 1e-5)

    def test_generator(self):

        step = 6

        style = [
            torch.randn(2, 512),
            torch.randn(2, 512),
            torch.randn(2, 512),
            torch.randn(2, 512),
            torch.randn(2, 512),
            torch.randn(2, 512),
            torch.randn(2, 512),
        ]

        noise = []

        for i in range(step + 1):
            size = 4 * 2 ** i
            noise.append(torch.randn(2, 1, size, size))

        # g1 = builder3.build(["input_1", "input_2", "input_3"], "alpha_mix_3")
        b = GeneratorBuilder(alpha=alpha)
        g1 = b.build(step)

        fake1 = g1.forward({
            "input": noise[0],
            "style1": style[0],
            "style2": style[1],
            "style3": style[2],
            "style4": style[3],
            "style5": style[4],
            "style6": style[5],
            "style7": style[6],
            "noise1": noise[0],
            "noise2": noise[1],
            "noise3": noise[2],
            "noise4": noise[3],
            "noise5": noise[4],
            "noise6": noise[5],
            "noise7": noise[6]
        })["out_rgb7"]

        g = Generator(512)
        g.progression[0] = b.builder.nodes["progression_1"].module
        g.progression[1] = b.builder.nodes["progression_2"].module
        g.progression[2] = b.builder.nodes["progression_3"].module
        g.progression[3] = b.builder.nodes["progression_4"].module
        g.progression[4] = b.builder.nodes["progression_5"].module
        g.progression[5] = b.builder.nodes["progression_6"].module
        g.progression[6] = b.builder.nodes["progression_7"].module
        g.to_rgb[0] = b.builder.nodes["to_rgb_1"].module
        g.to_rgb[1] = b.builder.nodes["to_rgb_2"].module
        g.to_rgb[2] = b.builder.nodes["to_rgb_3"].module
        g.to_rgb[3] = b.builder.nodes["to_rgb_4"].module
        g.to_rgb[4] = b.builder.nodes["to_rgb_5"].module
        g.to_rgb[5] = b.builder.nodes["to_rgb_6"].module
        g.to_rgb[6] = b.builder.nodes["to_rgb_7"].module

        fake = g.forward(style, noise, step, alpha)

        self.assertTrue((fake - fake1).abs().max().item() < 1e-5)

    def test_discriminator(self):

        b = DiscriminatorBuilder(alpha=alpha)
        b.builder.plot("disc")
        disc = b.build(3)
        print(disc.from_names)
        images = torch.randn(5, 3, 32, 32)
        res = disc.forward({"rgb4": images})
        print(res['out_pr0'].shape)

