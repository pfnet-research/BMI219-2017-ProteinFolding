import chainer
from chainer import functions as F
from chainer import links as L

from lib.models import mlp as mlp_
from lib.models import cnn as cnn_
from lib.models import rnn as rnn_


class Model(chainer.Chain):

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        for l in self.children():
            l.train = val
        self._train = val

    def __call__(self, acid, profile):
        timestep = acid.data.shape[1]
        acid = self.embed(acid)
        x = F.concat((acid, profile), axis=2)
        x = F.expand_dims(x, 1)
        x = F.relu(self.cnn(x))
        xs = F.split_axis(x, timestep, 2)
        xs = self.rnn(xs)
        ys_structure = F.stack([self.fc_structure(x) for x in xs], -1)
        ys_absolute_solvent = F.hstack([self.fc_absolute_solvent(x)
                                        for x in xs])
        ys_relative_solvent = F.hstack([self.fc_relative_solvent(x)
                                        for x in xs])
        return ys_structure, ys_absolute_solvent, ys_relative_solvent


def make_model(vocab, embed_dim, channel_num,
               rnn_dim, fc_dim, structure_class_num):
    embed = L.EmbedID(vocab, embed_dim, ignore_label=-1)

    channels = (channel_num,) * 3
    kernel_size = embed_dim + vocab
    windows = [(3, kernel_size), (7, kernel_size), (11, kernel_size)]
    cnn = cnn_.MultiScaleCNN(1, channels, windows)

    conv_out_dim = channel_num * 3
    rnn = rnn_.StackedBiRNN(conv_out_dim, rnn_dim, rnn_dim, 3)

    fc_structure = mlp.MLP(fc_dim, structure_class_num)
    fc_absolute_solvent = mlp.MLP(fc_dim, 1)
    fc_relative_solvent = mlp.MLP(fc_dim, 1)

    model = Model(embed=embed, cnn=cnn, rnn=rnn,
                  fc_structure=fc_structure,
                  fc_absolute_solvent=fc_absolute_solvent,
                  fc_relative_solvent=fc_relative_solvent)
    model.train = True
    return model
