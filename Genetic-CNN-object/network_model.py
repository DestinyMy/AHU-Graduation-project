import torch.nn as nn
from Decode import ResidualDecoder


class Network(nn.Module):
    """
    Entire network.
    Made up of Phases.
    """
    def __init__(self, genome, channels, repeats=None):
        """
        Network constructor.
        :param genome: depends on decoder scheme, for most this is a list.
        :param channels: list of desired channel tuples.
        :param out_features: number of output features.
        :param decoder: string, what kind of decoding scheme to use.
        """
        super(Network, self).__init__()

        assert len(channels) == len(genome), "Need to supply as many channel tuples as genes."
        if repeats is not None:
            assert len(repeats) == len(genome), "Need to supply repetition information for each phase."

        self.model = ResidualDecoder(genome, channels, repeats).get_model()

        #
        # After the evolved part of the network, we would like to do global average pooling and a linear layer.
        # However, we don't know the output size so we do some forward passes and observe the output sizes.
        #

        self.gap = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        # We accumulated some unwanted gradient information data with those forward passes.
        self.model.zero_grad()

    def forward(self, x):
        """
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        """
        x = self.gap(self.model(x))
        out = self.layer(x)

        return out
