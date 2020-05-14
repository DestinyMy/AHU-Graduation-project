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
        self.fc = nn.Linear(7*7*self.channels[2][1], num_classes)

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
        if index != 2:
            layer = nn.Sequential(
                nn.Conv2d(self.channels[index+1][0], self.channels[index+1][1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.channels[index+1][1])
            )
        else:
            layer = Identity()
            # layer = nn.AvgPool2d(8)
        return layers, layer.to(device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        for i in range(3):
            [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7], self.layer = self.get_cell(self.genotype, i)
            residual = out
            out1 = self.relu(self.l1(out))
            out = self.relu(self.l2(out1))
            out1 = self.relu(self.l3(out1)) + self.relu(self.l4(out))
            out = self.relu(self.l5(out)) + self.relu(self.l6(out1))
            out = self.l7(out)
            out += residual
            out = self.relu(out)
            out = self.layer(out)
            if i != 2:
                out = self.relu(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def assemble(genotype, channels, num_classes):
    model = Model(genotype, channels, num_classes)
    return model
