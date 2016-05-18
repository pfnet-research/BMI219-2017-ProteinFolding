import chainer
from chainer import functions as F
from chainer import links as L


class MLP(chainer.ChainList):

    def __init__(self, *units):
        super(MLP, self).__init__(*[
            L.Linear(None, unit) for unit in units])
        self.train = True

    def __call__(self, x):
        N = len(self)
        for i, f in enumerate(self):
            x = f(x)
            if i + 1 != N:
                x = F.relu(F.dropout(x, train=self.train))
        return x
