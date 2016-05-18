import chainer
from chainer import functions as F
from chainer import links as L
import six


class MultiScaleCNN(chainer.ChainList):

    def __init__(self, in_channel, channel_nums, windows):
        assert len(channel_nums) == len(windows)

        convolutions = [L.Convolution2D(in_channel, c, w, pad=(w[0] // 2, 0))
                        for c, w in six.moves.zip(channel_nums, windows)]

        super(MultiScaleCNN, self).__init__(*convolutions)

    def __call__(self, x):
        xs = [l(x) for l in self]
        return F.concat(xs, 1)
