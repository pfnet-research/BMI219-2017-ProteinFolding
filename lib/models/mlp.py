import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.ChainList):

    def __init__(self, *units):
        super(MLP, self).__init__(*[
            L.Linear(None, unit) for unit in units])
        self.train = True

    def __call__(self, x):
        for l in self[:-1]:
            x = F.relu(l(x))
        return self[-1](x)
