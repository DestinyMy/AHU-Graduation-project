import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self, genotype, channels,  num_classes=10):
        super(Model, self).__init__()
        self.genotype = genotype
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channels[0][0], self.channels[0][1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels[0][1]),
        )
        self.relu = nn.ReLU(inplace=True)
        [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7] = [None, None, None, None, None, None, None]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

    def get_cell(self, genotype, index):
        layers = []
        for geno in genotype:
            if geno == '1*1 convolution':
                layers.append(nn.Sequential(
                    nn.Conv2d(self.channels[index][1], self.channels[index][1], kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(self.channels[index][1])).to(device))
            elif geno == '3*3 convolution':
                layers.append(nn.Sequential(
                    nn.Conv2d(self.channels[index][1], self.channels[index][1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.channels[index][1])).to(device))
            else:
                layers.append(Identity().to(device))
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        for i in range(4):
            [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7] = self.get_cell(self.genotype, i)
            residual = out
            out1 = self.relu(self.l1(out))
            out = self.relu(self.l2(out1))
            out1 = self.relu(self.l3(out1)) + self.relu(self.l4(out))
            out = self.relu(self.l5(out)) + self.relu(self.l6(out1))
            out = self.l7(out)
            out += residual
            out = self.relu(out)
            out = self.max_pool(out)
        out = self.layer(out)
        return out


def assemble(genotype, channels):
    model = Model(genotype, channels)
    return model
