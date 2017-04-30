import chainer
from chainer import functions as F
from chainer import links as L
import six

from lib.models import gru


class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim):
        forward = chainer.ChainList(
            gru.GRU(input_dim, hidden_dim),
            gru.GRU(hidden_dim * 2, hidden_dim),
            gru.GRU(hidden_dim * 2, hidden_dim))
        reverse = chainer.ChainList(
            gru.GRU(input_dim, hidden_dim),
            gru.GRU(hidden_dim * 2, hidden_dim),
            gru.GRU(hidden_dim * 2, hidden_dim))
        super(StackedBiRNN, self).__init__(
            forward=forward, reverse=reverse)
        self.train = True

    def reset_state(self):
        for l in self.forward:
            l.reset_state()
        for l in self.reverse:
            l.reset_state()

    def __call__(self, xs, is_seq):
        is_seq_t = is_seq.T
        self.reset_state()
        N = len(self.forward)
        T = len(xs)
        for i, (f, r) in enumerate(six.moves.zip(self.forward, self.reverse)):
            xs_f = [f(x, seq) for x, seq in six.moves.zip(xs, is_seq_t)]
            xs_r = [r(xs[i], is_seq[i]) for i in six.moves.range(T - 1, -1, -1)]
            xs_r.reverse()
            xs = [F.concat((x_f, x_r))
                  for (x_f, x_r) in six.moves.zip(xs_f, xs_r)]
            if i + 1 != N:
                xs = [F.dropout(x, train=self.train) for x in xs]
        return xs
