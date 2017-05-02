# Introduction

Prediction of the secondary structure of proteins is classical, but still important
and active research area in bioinformatics.
In this example, we apply deep learning to solve this task.
We will adopt the architecture that recently proposed in [1], which combines RNNs and CNNs.

# Dataset creation

We use the same dataset as the original paper, which is available from
the following URL:

* Download URL: http://www.princeton.edu/~jzthree/datasets/ICML2014/

Fortunately, the creator of the dataset, (which is different from the authors of [1])
had preprocessed the raw dataset to some extent already.
So, we need the minimal preprocessing to plug the data to our model.

Each sample represents a single protein and has the following information:

* Amino acid sequence
* Structure information
* Profile features obtained from the PSI-BLAST log file [1]
* Solvenct accesibility (relative and absolute, respectively).

We use the amino acid sequence as a input feature vector
and the structure information as a target label.

In the original paper, profile features are also used as an input feature,
to do a multi-task learning.

Further, in [1], they predict solvency information as well as structure information during training to enhance the prediction accuracy of the original task.
But to simplify the example, we only use protein ID and does not use structure information.

Amino acid sequences are represented as a sequence of integers,
each of which represents an ID of one of 21 amino acids.

The structure is represented as Q8, or one of the 8 categories:
3_10−helix (G), α−helix (H), π−helix (I), β−strand (E), β−bridge (B), β−turn (T), bend (S) and loop or irregular (L).
This information is attached to each amino acid.
Therefore, the length of amino acid sequence and that of structure information.

The original sequence length is 700.
But to reduce the computational time, we cut off the first 100 sequences.

Note that not all sequences have 700 amino acids,
if the length is less than 700, we pad the special integer that represents "no sequence".

In summary, it is a sequential 8-class classification as a machine learning task.

## Tuple dataset

We prepare the dataset as an instance of [`TupleDataset`](http://docs.chainer.org/en/stable/reference/datasets.html#chainer.datasets.TupleDataset):

```python
def load(fname, V, C, T=700, L=57):
    raw_data = numpy.load(fname)

    '''
    extract amino acid sequence and structure labels from raw data
    '''

    return datasets.TupleDataset(acids, structure_labels)
```


# Model

## overall

The model consists of four sub networks: ID embedding,
(multi-scale) convolutional neural network (CNN),
(bi-directional) recurrent neural network (RNN),
and multi-layer perceptron (MLP).

In this example, we construct the model in `lib.models.model.make_model`.
This method essentially constructs each component as a chain
and builds the whole model with them.

```python
def make_model(vocab, embed_dim, channel_num,
               rnn_dim, fc_dim, class_num):
    # ..
    embed = L.EmbedID(vocab, embed_dim, ignore_label=-1)
    # ...
    cnn = cnn_.MultiScaleCNN(1, channels, windows)
    # ...
    rnn = rnn_.StackedBiRNN(conv_out_dim, rnn_dim)
    # ...
    mlp = mlp_.MLP(mlp_dim, class_num)
    # ...
    model = Model(embed=embed, cnn=cnn, rnn=rnn, mlp=mlp)
    # ...
    return model
```

Forward propagation is defined in `Model.__call__` as is customarily done in Chainer.
It sequentially applies input features to each component.

```python
class Model(chainer.Chain):

    # (Omitted)

    def __call__(self, x):
        timestep = x.data.shape[1]
        x = self.embed(x)  # apply ID embedding
        x = F.expand_dims(x, 1)
        x = F.relu(self.cnn(x))  # apply CNN
        xs = F.split_axis(x, timestep, 2)
        xs = self.rnn(xs)  # apply RNN
        ys = [self.mlp(x) for x in xs]  # apply MLP
        return F.stack(ys, -1)
```

In the following section, we explain the detail of each component one by one.

## ID embedding

`L.EmbedID` converts each "word" represented as integer ID to a trainable vector.
This is operation is called "embedding" as we embed each word to a feature space.

In natural language processing research, we sometimes handle sentences,
which is a sequence of words.
Usually, each are represented as IDs, which are discrete integers rather than
continuous values like
In analogy, proteins are represented as "biological sentences" which consists of
21 types of amid acids ("biological word").

Each protein is represented as integers.

In usual setting, IDs are converted to one-hot vector and fed to
e.g. fully-connected layer.
Let the number of protein types be `C`, then,
the first fully connected layer has a weight matrix of size `C x D`
where `D` is the number of output units.

`L.EmbedID` converts an integer to one-hot vector and applies it to a fully-connected layer.
For computational efficiency, the implementation takes a different approach, but, it does essentially the same thing.

If `B` is a batch size and `T` is a length of each sample, the input to the model has a shape `(B, T)`. The shape of the output is `(B, T, D)` where `D` is a embedding dimension.

# CNN

We next convolve the resulting embedded vector along time direction.
As each sample has a shape `(T, D)` where `T` is a time step
and `D` is the dimension of embedded vectors,
we convolve the vectors with the kernel of shape `(t, D)`.

Following [1], we use three types of kernels of different lengths `t`.

```python
cnn = cnn_.MultiScaleCNN(
    1,  # input channel
    [64, 64, 64],  # output channels (64 kernels for each kernel size)
    [(3, embed_dim), (7, embed_dim), (11, embed_dim)])  # kernel sizes
```

As `L.Convolution2D` cannot handle kernels of different sizes in single chain.
Therefore, we prepare as many chains as the types of kernels.
Specifically, we create `L.Convolution2D` for kernels of length 3, 7, and 11 and
each chain has 64 kernels, resulting in 192 kernels in total.

```python
class MultiScaleCNN(chainer.ChainList):

    def __init__(self, in_channel, out_channels, kernel_sizes):
        convolutions = [L.Convolution2D(in_channel, c, k, pad=(k[0] // 2, 0))
                        for c, k in six.moves.zip(out_channels, kernel_sizes)]
        super(MultiScaleCNN, self).__init__(*convolutions)
```

Note that we add pad to keep the output size same as the input.

The forward propagation is to push input vectors to each links and concatenate
the outputs along the channel axis.
(`ChainList` can get child links via `self`)


```python
class MultiScaleCNN(chainer.ChainList):

    def __call__(self, x):
        xs = [l(x) for l in self]
        return F.concat(xs, 1)
```

Q. Check the shape of input and output tensors of `MultiScaleCNN`.


# RNN

## Stateless and stateful GRUs

Following the original paper, we use GRU as a RNN unit, which is one of the most
common building blocks of RNNs as well as with LSTM.

Roughly speaking, there are two ways to implement RNN units: stateless and stateful.
Stateless RNN units does not hold the internal state and takes it as an input.

At time `t`, the status update is formulated as follows:

```
h' = f(h, x)
```

where `x` is a input at time `t` and `h'` and 'h' are states of RNN at time `t-1` and `t`, respectively.

The pseudo code of Stateless GRU is as follows:

```python
class StatelessGRU(object):

    def __call__(self, h, x):
        return f(h, x)
```

Contrary to that, the stateful GRU holds the state of RNN and update it
at every step:

```python
class StatefulGRU(object):

    def __call__(self, x):
        self.h = f(self.h, x)
        return self.h
```

Some deep learning frameworks supports either of stateless and stateful and some supports both.
Chainer implements a stateless GRU as [L.GRU](http://docs.chainer.org/en/stable/reference/links.html?highlight=GRU#chainer.links.GRU)
and a stateful one as [L.StatefulGRU](http://docs.chainer.org/en/stable/reference/links.html?highlight=GRU#chainer.links.StatefulGRU)


## Build stacked bi-directional GRUs

We use two-layered bi-directional GRU.
First, we create a function that construct a uni-directional RNN.

```python
def make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num):
    grus = [L.StatefulGRU(input_dim, hidden_dim)]
    grus.extend([L.StatefulGRU(hidden_dim * 2, hidden_dim)
                 for _ in range(layer_num - 2)])
    grus.append(L.StatefulGRU(hidden_dim * 2, out_dim))
    return chainer.ChainList(*grus)
```

Q. Why the output of layers other than the first layer is doubled?

We combine two uni-directional RNNs, one is  going forward
to create a bi-directional RNN:

I did not use "reverse" instead of "backward" because what it implements is actually forward propagation :).

```python
class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, out_dim, layer_num):
        forward = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        reverse = make_stacked_gru(input_dim, hidden_dim, out_dim, layer_num)
        super(StackedBiRNN, self).__init__(
            forward=forward, reverse=reverse)
```

As we will insert a dropout, which should behave differently in training and test phases, to the model. We add an attribute `train`, to specifies the mode of the model:

```python
class StackedBiRNN(chainer.Chain):

    def __init__(self, input_dim, hidden_dim, out_dim, layer_num):
        # (Omitted)
        self.train = True
```

Finally, its forward propagation is defined in `__call__`:

```python
class StackedBiRNN(chainer.Chain):

    # (Omitted)

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

Here, `xs_f` is a list of the output of the forward RNN and `xs_r` is the reverse RNN.
Be aware that we have to feed the reverse RNN with the input data from the last.
`[::-1]` is a handy way to reverse a sequence:

```python
a = [1, 2, 3]
a[::-1] # => [3, 2, 1]
```

Q. Check [the specification of slices](https://docs.python.org/3/library/functions.html#slice) to see how it works.

As we expect `StackedBiRNN` is followed by a MLP, we insert dropout layers
to not only the intermediate layers but to the final layer.


Q. Strictly speaking, we must skip the update of internal states when the input at corresponding time is `no_seq`. But we do not do it for simplicity.
See "skip-status-update" branch if you are interested in how to do that.

## MLP

[1] used 2 fully-connected layers to get the final prediction.
There are two possibilities how to put them on top of RNN layers.
So far as I read the original paper, I could not find out which method was adopted.

One way is to bundle the RNN outputs along all time steps and feed a huge
multi-layer perceptron(MLP) with them. This approach is simple.
But as the number of output units, which is time step multiplied by unit numbers for each time step, to the fully-connected layer is fixed throughout the whole dataset,
the time step of each sample should be same.
This could not be problematic in this example because the training and testing dataset
is preprocessed so that all samples have same length.

The other way is to apply the same MLP to each time step.
In this case, we do not have such a restriction on time length.
We choose this method in this example.

We should simply apply the same to the output sequence

```python
class Model(chainer.Chain):

    # (Omitted)

    def __call__(self, x):
        # (Omitted)
        ys = [self.mlp(x) for x in xs]  # apply MLP
        return F.stack(ys, -1)
```

Q. We can implement the MLP part equivalent to the former one with
[`L.MLPConvolution2D`](http://docs.chainer.org/en/stable/reference/links.html?highlight=MLPconvolution2D#chainer.links.MLPConvolution2D).
Read the document and re-implement `Model.__call__`.

## Putting them all together

Q. In the original paper, the model has a direct connection from the output of CNN to
the input of MLP, which our current model does not have. Implement it so that the architecture
agree with the original one.
Looking at their experiment, it does not improve the final performance so much.
But it is a good excersize to get used to Chainer.

# Training

## Weight decay

*Weight decay* is a common method to regularize deep learning models.
In each parameter update, we shrink the parameters as follows:

```
w <- w - \eta w
```
where `eta` is a hyper parameter that determines the amount of regularization.

It is equivalent to online L2 regularization.

In Chainer, we can realize weight decay as a [`WeightDecay`](http://docs.chainer.org/en/stable/reference/core/optimizer.html?highlight=WeightDecay#chainer.optimizer.WeightDecay)
hook to optimizers.

The hook is any callable that takes the hooked optimizer in its `__call__` method.

We apply weight decay, following the original paper [1].

```python
optimizer.add_hook(WeightDecay(1e-3))
```


## TestEvaluator

As the model includes dropout, which should behave differently in training and test phases, we need to change it accordingly.

The core part of [`Evaluator`](http://docs.chainer.org/en/stable/reference/extensions.html#evaluator) is
`evaluate`. So we overwrite it as follows:

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

We will release Chainer v2, or the first major version up of Chainer soon.

In v2, Chainer has the current phases (training/test etc.) as a global variable.
So, each chain need not to have an attribute to determine its mode by themselves
(e.g. `train` attribute of `Model` in this example).

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

# Multi-task, multi-modal learning

Q. As explained earlier, we do not use profile features and solvent accesibility.
Change the code to support multi-task, multi-modal learning

1. Fix `load` so that `TupleDataset` includes all information.
2. The model is wrapped with `L.Classifier`, but `__call__` method of [`L.Classifier`](http://docs.chainer.org/en/stable/reference/links.html#classifier) interprets that its last argument as the target label and the rest as input features. This is not the case in this extension. So, create customized `Classifier` and substitute it with `L.Classifier`.
3. Fix `Model` to accept multiple inputs.


# Reference
[1] Li, Z., & Yu, Y. (2016). Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. arXiv preprint arXiv:1604.07176.
