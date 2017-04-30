import chainer
from chainer import cuda
from chainer import links as L
from chainer import functions as F


class GRU(L.StatefulGRU):

    def __call__(self, x, is_seq):
        h_old = getattr(self, 'h', None)
        h = super(GRU, self).__call__(x)
        xp = cuda.get_array_module(x)
        if h_old is None:
            h_old = chainer.Variable(xp.zeros_like(self.h, dtype=xp.float32), volatile='auto')
        is_seq = xp.broadcast_to(is_seq[..., None], self.h.shape)
        self.h = F.where(is_seq, h, h_old)
        return self.h
