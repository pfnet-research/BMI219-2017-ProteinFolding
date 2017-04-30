import chainer
from chainer import functions as F
from chainer import links as L
import six


class MultiScaleCNN(chainer.ChainList):

    def __init__(self, in_channel, out_channels, kernel_sizes):
        assert len(out_channels) == len(kernel_sizes)

        convolutions = [L.Convolution2D(in_channel, c, k, pad=(k[0] // 2, 0))
                        for c, k in six.moves.zip(out_channels, kernel_sizes)]

        super(MultiScaleCNN, self).__init__(*convolutions)

    def __call__(self, x):
        xs = [l(x) for l in self]
        return F.concat(xs, 1)
