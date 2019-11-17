from torch import nn
import random
from model import PixelNorm, EqualLinear


class StyleNoise(nn.Module):

    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def sample(self, input, step, mean_style=None, style_weight=0, mixing_range=(-1, -1)):

        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        styles_stack = []

        if len(styles) < 2:
            inject_index = [10]

        else:
            inject_index = random.sample(list(range(step)), len(styles) - 1)

        crossover = 0

        for i in range(step + 1):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(styles))

                style_step = styles[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = styles[1]

                else:
                    style_step = styles[0]

            styles_stack.append(style_step)

        return styles_stack
