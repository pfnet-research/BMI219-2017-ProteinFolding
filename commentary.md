# Introduction

Prediction of the secondary structure of proteins is classical, but still important
and active research area in bioinformatics.
In this example, we employ deep learning to tackle a protein-folding problem.

We will adopt the architecture that recently proposed in [1], which combines RNNs and CNNs
with a modification to simplify the implementation.
If you are interested in we will construct a full model in questions.

# Dataset

## Retrieval

We use the same dataset as the original paper, which is available from
the following URL:

* Download URL: http://www.princeton.edu/~jzthree/datasets/ICML2014/

Fortunately, the creator of the dataset, (which is different from the authors of [1])
had preprocessed the raw dataset to some extent already.
So, we need the minimal preprocessing to plug the data to our model.

## Problem formulation

Each sample represents a single protein and has the following information:

* Amino acid sequence
* Structure information
* Profile features obtained from the PSI-BLAST log file [1]
* Solvent accessibility (relative and absolute).

The goal is to construct a model that predicts the structure information from the others.
To do that, they apply multi-task and multi-modal learning.
Specifically, they construct a model that predicts the structure information *and*
the solvent accessibility from the amino acid sequence and the profile features.
(They use solvent accessibility for only training and the accuracy is evaluated how accurately the structure information is modeled).
But to keep the example simple, we used the amino acid sequences as input features
and the model does not predict solvent accessibility.

## Data specification

We formulate the problem as a sequential 8-class classification task.

The amino acid sequence is represented as an 1-dimensional array of integers,
each of which represents the ID of one of 21 amino acids.

The structure information is attached to each amino acid,
which is represented as Q8, or one of the 8 categories:
3_10−helix (G), α−helix (H), π−helix (I), β−strand (E), β−bridge (B), β−turn (T), bend (S) and loop or irregular (L).
Therefore, the length of these two sequences are same.

The sequence length of the original dataset is 700.
Note that not all sequences have 700 amino acids.
If the length is less than 700, we pad the special (or dummy) integer that represents "no sequence" (-1 in this example).
To the contrary, if it is longer, they truncate the first 700 acids.
But to reduce the computational time, we use the first 100 subsequences in this example.

We prepare the dataset as an instance of [`TupleDataset`](http://docs.chainer.org/en/stable/reference/datasets.html#chainer.datasets.TupleDataset):


# Model

## Overall

The model consists of four subnetworks: Word embedding,
multi-scale Convolutional Neural Network (CNN),
(bi-directional) Recurrent Neural Network (RNN),
and Multi-Layer Perceptron (MLP).

We can get the model with `lib.models.model.make_model`:
Here is the pseudo code:
```python
def make_model(vocab, embed_dim, channel_num,
               rnn_dim, mlp_dim, class_num):
    # (Omitted)
    embed = L.EmbedID(...)
    cnn = cnn_.MultiScaleCNN(...)
    rnn = rnn_.StackedBiRNN(...)
    mlp = mlp_.MLP(...)
    model = Model(embed=embed, cnn=cnn, rnn=rnn, mlp=mlp)
    # (Omitted)
    return model
```
This method essentially constructs each component as a "chain"
(i.e. an instance of [chainer.Chain](http://docs.chainer.org/en/stable/reference/core/link.html#chainer.Chain))
and builds the whole model with them.


Forward propagation is defined in `Model.__call__` as is customarily done in Chainer.
It sequentially applies input features to each component:

```python
class Model(chainer.Chain):

    # (Omitted)

    def __call__(self, x):
        timestep = x.data.shape[1]
        x = self.embed(x)  # apply Word embedding
        x = F.expand_dims(x, 1)
        x = F.relu(self.cnn(x))  # apply CNN
        xs = F.split_axis(x, timestep, 2)
        xs = self.rnn(xs)  # apply RNN
        ys = [self.mlp(x) for x in xs]  # apply MLP
        return F.stack(ys, -1)
```

In the following section, we explain the detail of each component one by one.

## Word embedding

First, we convert amino acid sequences to trainable float vectors.

When we handle sequences of IDs, typically we convert the sequences into
*one-hot vectors* and feed a fully-connected layer with them.
[`L.EmbedID`](http://docs.chainer.org/en/stable/reference/links.html#embedid) is responsible for this job in Chainer.
Precisely speaking, the implementation takes a different approach for computational efficiency, but it does essentially the same thing.

Let `B` be a batch size and `T` a length of each sample.
Then, the input to the model has a shape `(B, T)`.
The shape of the output is `(B, T, D)` where `D` is a embedding dimension.

The conversion of ID (discrete variables) to trainable vectors is sometimes called
*word embedding* especially in NLP (natural language processing) fields,
in which they analyze sentences, or sequences of words.
In analogy, we can think of proteins as "biological sentences" consisting of
21 types of amid acids ("biological word").
The term "embedding" comes from the intuition of embedding each ID to a feature space.


## Convolutional Neural Networks (CNN)

Next we convolve the resulting embedded vectors along the sequence.
As each sample has a shape `(T, D)` where `T` is a time step and `D` is the dimension of embedded vectors, filters will have a shape of the form `(t, D)`.

Following [1], we use three types of filters of different `t`.
Specifically, we create 64 filters of length 3, 7, and 11, respectively,
resulting in 192 filters in total:

```python
cnn = cnn_.MultiScaleCNN(
    1,  # input channel
    [64, 64, 64],  # output channels (64 kernels for each kernel size)
    [(3, embed_dim), (7, embed_dim), (11, embed_dim)])  # kernel sizes
```

As [`L.Convolution2D`](http://docs.chainer.org/en/stable/reference/links.html#convolution2d) cannot handle filters of different sizes in single chain.
Therefore, we prepare as many chains as the different filter shapes:

```python
class MultiScaleCNN(chainer.ChainList):

    def __init__(self, in_channel, out_channels, kernel_sizes):
        convolutions = [L.Convolution2D(in_channel, c, k, pad=(k[0] // 2, 0))
                        for c, k in six.moves.zip(out_channels, kernel_sizes)]
        super(MultiScaleCNN, self).__init__(*convolutions)
```

Note that we add padding to keep the length of the output same as that of the input.

The forward propagation is to put input vectors to each links and concatenate
the outputs along the channel axis.
We implement the forward propagation in `__call__` as usual:

```python
class MultiScaleCNN(chainer.ChainList):

    def __call__(self, x):
        xs = [l(x) for l in self]
        return F.concat(xs, 1)
```

Note that we can have access to child links via `ChainList.self`.

Q. Check the shape of input and output tensors of `MultiScaleCNN`.


## Recurrent Neural Networks (RNN)

Following the original paper, we use GRU as a RNN unit, which is one of the most
common building blocks of RNNs as well as LSTM.

### Stateless and stateful RNNs

Roughly speaking, there are two ways to implement RNN units: stateless and stateful.
Suppose the status update at time `t` is formulated as follows :

```
h' = update(h, x)
```

Here, `h'` and `h` are states of RNN at time `t-1` and `t`, respectively, `x` is a input at time `t`, and `update` is a update formula specific to each RNN unit.

Stateless RNN units does not hold its internal state inside.
Alternatively it takes the state as an input.
The pseudo code of the stateless GRU is as follows:

```python
class StatelessGRU(object):

    def __call__(self, h, x):
        return update_formula_of_gru(h, x)
```

On the other hand, the stateful RNN holds its state and update it
at every step:

```python
class StatefulGRU(object):

    def __call__(self, x):
        self.h = update_formula_of_gru(self.h, x)
        return self.h
```

Some deep learning frameworks supports either of stateless and stateful RNNs and some supports both.
Chainer implements a stateless GRU as [L.GRU](http://docs.chainer.org/en/stable/reference/links.html?highlight=GRU#chainer.links.GRU)
and a stateful one as [L.StatefulGRU](http://docs.chainer.org/en/stable/reference/links.html?highlight=GRU#chainer.links.StatefulGRU)


### Build stacked bi-directional GRUs

We use two-layered bi-directional GRU.
First, we create a function that constructs a uni-directional RNN.

```python
def make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num):
    grus = [L.StatefulGRU(input_dim, hidden_dim)]
    grus.extend([L.StatefulGRU(hidden_dim * 2, hidden_dim)
                 for _ in range(layer_num - 2)])
    grus.append(L.StatefulGRU(hidden_dim * 2, out_dim))
    return chainer.ChainList(*grus)
```

Q. Why the output of layers other than the first layer is doubled?

We combine two uni-directional RNNs, one is for going forward and the other going reverse, to create a bi-directional RNN:

```python
class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, out_dim, layer_num):
        forward = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        reverse = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        super(StackedBiRNN, self).__init__(
            forward=forward, reverse=reverse)
```
I use the term "reverse" instead of "backward" because what is implemented is actually forward propagation :).

As we will insert a dropout, which should behave differently in training and test phases, to the model. We put an attribute `train` to specifies the mode of the model:

```python
class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, out_dim, layer_num):
        # (Omitted)
        self.train = True
```

Finally, the forward propagation is defined in `__call__`:

```python
class StackedBiRNN(chainer.Chain):

    def __call__(self, xs):
        # (Omitted)
        for f, r in zip(self.forward, self.reverse):
            xs_f = [f(x) for x in xs]
            xs_r = [r(x) for x in xs[::-1]]
            xs_r.reverse()
            xs = [F.dropout(F.concat((x_f, x_r)), train=self.train)
                  for (x_f, x_r) in zip(xs_f, xs_r)]
        return xs
```

Here, `xs_f` is a list of the outputs from the forward RNN and `xs_r` is from the reverse RNN.
Be aware that we need to feed the reverse RNN with the input data in reverse order.
`[::-1]` is a handy way to reverse a sequence:

```python
a = [1, 2, 3]
a[::-1] # => [3, 2, 1]
```

As we expect that `StackedBiRNN` is followed by a MLP, we insert dropout layers
to not only the intermediate layers but also to the final layer.

Q. Check [the official document of slices](https://docs.python.org/3/library/functions.html#slice) to see how the reverse of lists explained above works.

Q. As we explained above, not all proteins have a length 700.
Therefore strictly speaking, we must skip the update of internal states when the input at corresponding time is a dummy character. We implement this modification in "skip-status-update" branch. Check the branch if you are interested in.

## Multi-layer perceptrons (MLP)

### Two possibilities of implementation

The original paper explains they used 2 fully-connected layers to get the final prediction.
There are two possibilities how to use them on top of RNN layers.
One possibility is to bundle the RNN outputs along all time steps and feed a huge
multi-layer perceptron(MLP) with them.
The other one is to apply the same MLP to each time step.
In this case, we do not have such a restriction on time length.
I could miss some information, but so far as I read the paper, I could not find out which method was adopted.
We choose the second approach in this example.

(You can skip the latter half of this section)

The first approach is intuitive.
But the length of each sample in the dataset must be same because the input dimension of the MLP (the length of the output of the RNN times the dimension of the output at each step) is fixed.
In this particular example, this could not be problematic in this example because the training and testing dataset is preprocessed so that all samples have same length.
But in general samples in the dataset could have different lengths.
In that case, the first approach.


### Implementation

The implementation is quite simple.
We should simply apply the same MLP to the output sequence:

```python
class Model(chainer.Chain):

    def __call__(self, x):
        '''
        Apply ID embedding, CNN, RNN sequentially to x to get xs.
        xs is a list of variables of shape `(D,)`.
        '''
        ys = [self.mlp(x) for x in xs]  # apply MLP
        return F.stack(ys, -1)
```

Q. We can implement the MLP part with
[`L.MLPConvolution2D`](http://docs.chainer.org/en/stable/reference/links.html#chainer.links.MLPConvolution2D).
Originally, this chain is made to realize NetworkInNetwork (NIN) architecture, which is used in the computer vision field. But we can also use it in this example.
Read the document and re-implement `Model.__call__`.


## Putting them all together

Q. In the original paper, the model has a direct connection from the output of CNN to
the input of MLP, which our current model does not have.
Implement it so that the architecture agree with the original one.
Looking at their experiment, this change does not improve the final performance so much.
But it is a good exersize to get used to Chainer.


# Training

## Weight decay

*Weight decay* is a common method to regularize deep learning models.
In each parameter update, we shrink the parameters as follows:

```
w <- w - \eta w
```

where `eta` is a hyper parameter that determines the amount of regularization.

In Chainer, we can realize weight decay as a [`WeightDecay`](http://docs.chainer.org/en/stable/reference/core/optimizer.html?highlight=WeightDecay#chainer.optimizer.WeightDecay)
hook to optimizers. Following the original paper, we apply weight decay.

```python
optimizer.add_hook(WeightDecay(1e-3))
```

## TestEvaluator

As the model includes dropout, which should behave differently in training and test phases, we need to devise some trick to switch the "mode" of the architecture appropriately.

[`chainer.trainer.Evaluator`](http://docs.chainer.org/en/stable/reference/extensions.html#evaluator) provides a general way for evaluating the model at the testing phase.
The core part of the `Evaluator` is `evaluate` method.
So, we inherit `Evaluator` and overwrite the function to switch the mode of the model temporarily as follows:

```python
class Evaluator(E.Evaluator):

    def evaluate(self):
        predictor = self.get_target('main').predictor
        train = predictor.train
        predictor.train = False
        ret = super(Evaluator, self).evaluate()
        predictor.train = train
        return ret
```

### Note on Chainer v2

We will release Chainer v2, the first major version up of Chainer soon.
In the v2, Chainer has the current phases (training/test etc.) as a global variable,
`chainer.configuration.train`.
So, each chain need not to have an attribute to determine its mode by themselves
(like the `train` attribute of `Model` in this example).
Specifically, mode switching would be something like this
(be aware that v2 is currently under development, so APIs are subject to change):

```python
model = Model(embed=embed, cnn=cnn, rnn=rnn, mlp=mlp)

x = numpy.random.uniform(-1, 1, (10, 100))  # dummy minibatch

with chainer.config.using_config('train', True):
    y = model(x)  # runs in training mode

with chainer.config.using_config('train', False):
    y = model(x)  # runs in test mode
```

As we expect the Chainer v2 will be released at least after this course,
we implement and explain the code to work with Chainer v1.


# Multi-task, multi-modal learning

Q. As explained earlier, we do not use profile features and solvent accessibility.
We want to change the code to support multi-task, multi-modal learning as the original paper does.
I have implemented it in the "multi-modal-multi-task" branch.
Check the branch or change the code in the master branch by following these steps:

1. Fix `load` so that the dataset includes all information. One possible implementation is to make a [`TupleDataset`] consists of 5 elements (amino-acid, structure, profile, absolute solvent accessibility, and relative solvent accessibility).
2. The model is wrapped with [`L.Classifier`](http://docs.chainer.org/en/stable/reference/links.html#classifier), but currently, we calculate the loss value from only the structure prediction. Change `L.Classifier` or create a customized classifier so that it computes loss value from the (absolute/relative) solvent accessibility
(Hint: [`L.Classifier.__call__`](https://github.com/pfnet/chainer/blob/master/chainer/links/model/classifier.py#L43) assume that only the last argument as the target label.
This is not the case if each sample is a 5-tuples))
3. Fix `Model` to accept the structure information and the profile information and output not only the prediction of structure but also that of solvent accessibilities.


# Reference
[1] Li, Z., & Yu, Y. (2016). Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. arXiv preprint arXiv:1604.07176.
