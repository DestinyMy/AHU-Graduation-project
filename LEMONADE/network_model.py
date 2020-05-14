import torch
import torch.nn as nn
from Decode import ResidualDecoder


class Network(nn.Module):
    """
    Entire network.
    Made up of Phases.
    """
    def __init__(self, genome, channels, out_features, data_shape, repeats=None):
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

        out = self.model(torch.autograd.Variable(torch.zeros(1, channels[0][0], *data_shape)))
        shape = out.data.shape

        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1] * shape[2] * shape[3], out_features)

        # We accumulated some unwanted gradient information data with those forward passes.
        self.model.zero_grad()

    def forward(self, x):
        """
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        """
        x = self.gap(self.model(x))

        x = x.view(x.size(0), -1)

        return self.linear(x)
