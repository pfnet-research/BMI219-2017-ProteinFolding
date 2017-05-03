import argparse

import chainer
from chainer import cuda
from chainer import iterators
from chainer import links as L
from chainer import optimizer as optimizer_
from chainer import optimizers
from chainer import training
from chainer.training import extensions as E
import numpy

from lib.data import jzthree
from lib.evaluations import evaluator as evaluator_
from lib.models import model


parser = argparse.ArgumentParser(
    description='Protein second structure prediction')
# general
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--debug', action='store_true')
# IO
parser.add_argument('--train-file', type=str,
                    default='data/cullpdb+profile_6133_filtered.npy',
                    help='Path to training dataset')
parser.add_argument('--validate-file', type=str,
                    default='data/cb513+profile_split1.npy',
                    help='Path to validate dataset')
parser.add_argument('--out', default='result', type=str,
                    help='Path to the output directory')
# training parameter
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--epoch', default=100, type=int,
                    help='The number of training epoch.')
# model parameter
parser.add_argument('--embed-dim', type=int, default=50)
parser.add_argument('--channel-num', type=int, default=64)
parser.add_argument('--rnn-dim', type=int, default=600)
parser.add_argument('--fc-dim', type=int, default=1000)
args = parser.parse_args()

chainer.set_debug(args.debug)

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)
    chainer.cuda.get_device(args.gpu).use()

# data
V, C = 21, 8  # vocab, class_num
train_dataset = jzthree.load(args.train_file, V, C)
validate_dataset = jzthree.load(args.validate_file, V, C)
train_iter = iterators.SerialIterator(train_dataset, args.batchsize)
validate_iter = iterators.SerialIterator(validate_dataset, args.batchsize,
                                         repeat=False, shuffle=False)


# model
model = model.make_model(V, args.embed_dim, args.channel_num,
                         args.rnn_dim, args.fc_dim, C)
classifier = L.Classifier(model)
if args.gpu >= 0:
    classifier.to_gpu()

# optimizer
optimizer = optimizers.Adam()
optimizer.setup(classifier)
optimizer.add_hook(optimizer_.WeightDecay(1e-3))

# trainer
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

# extensions
log_report = E.LogReport(trigger=(10, 'iteration'))
print_report = E.PrintReport(['epoch', 'iteration',
                              'main/loss', 'main/accuracy',
                              'validation/main/loss',
                              'validation/main/accuracy',
                              'elapsed_time'])

evaluator = evaluator_.Evaluator(
    validate_iter, classifier, device=args.gpu)

trainer.extend(log_report)
trainer.extend(print_report)
trainer.extend(evaluator)
trainer.extend(E.ProgressBar(update_interval=1))

trainer.run()
