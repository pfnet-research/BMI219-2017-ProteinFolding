import chainer
from chainer import functions as F
from chainer import links as L
import six


def make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num):
    grus = [L.StatefulGRU(input_dim, hidden_dim)]
    grus.extend([L.StatefulGRU(hidden_dim * 2, hidden_dim)
                 for _ in range(layer_num - 2)])
    grus.append(L.StatefulGRU(hidden_dim * 2, out_dim))
    return chainer.ChainList(*grus)


class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, out_dim, layer_num):
        forward = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        reverse = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        super(StackedBiRNN, self).__init__(
            forward=forward, reverse=reverse)
        self.train = True

    def reset_state(self):
        for l in self.forward:
            l.reset_state()
        for l in self.reverse:
            l.reset_state()

    def __call__(self, xs):
        self.reset_state()
        for f, r in six.moves.zip(self.forward, self.reverse):
            xs_f = [f(x) for x in xs]
            xs_r = [r(x) for i in xs[::-1]]
            xs_r.reverse()
            xs = [F.dropout(F.concat((x_f, x_r)), train=self.train)
                  for (x_f, x_r) in six.moves.zip(xs_f, xs_r)]
        return xs
