# Introduction


# Dataset creation

We use the same dataset as [1].

The dataset is available from
http://www.princeton.edu/~jzthree/datasets/ICML2014/

Preprocess is already done by the original creator of the dataset.
We need the minimal preprocessing.

Each sample is represented as a pair of two sequences: one is a list of protein ids
and the other is a list of

So it is a kind of sequential prediction.

The original sequence length is 700.
To reduce the computational time, we cut off the first 100 sequences.

To simplify the example, we only use protein ID and does not use structure information.
Further, we do not do multi-task learning as does in [1].
In [1], they predict solvency information as well as structure information during training.

In "multi-modal" branch, we do multi-task, multi-modal learning with
by introducing the information that is ignored.


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
    model = Model(embed=embed, cnn=cnn, rnn=rnn, fc=fc)
    # ...
    return model
```

Forward propagation is defined in `Model.__call__` as is customarily done in Chainer.
It sequentially applies input features to each component.

```python
class Model(chainer.Chain):

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
def __call__(self, x):
    xs = [l(x) for l in self]
    return F.concat(xs, 1)
```

Q. Check the shape of input and output tensors of `MultiScaleCNN`.


# RNN

## Stateless and stateful GRUs

Following the original paper, we use GRU as a RNN unit, which is one of the most
common building blocks of RNNs as well as with LSTM.

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

`xs_f` is a list of the output of the forward RNN and `xs_r` is the reverse RNN.

Be aware that from the last

As we expect `StackedBiRNN` is followed by a MLP, we insert dropout layers
to not only the intermediate layers but to the final layer.


Strictly speaking, we must skip the update of internal states when the input at corresponding time is `no_seq`. But we do not do it for simplicity.
See "skip-status-update" branch if you are interested in how to do that.


# MLP convolution

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
We chose this method in this example.

```python
def __call__(self, x):
    timestep = x.data.shape[1]
    x = self.embed(x)
    x = F.expand_dims(x, 1)
    x = F.relu(self.cnn(x))
    xs = F.split_axis(x, timestep, 2)
    xs = self.rnn(xs)
    ys = [self.fc(x) for x in xs]
    return F.stack(ys, -1)
```

Chainer has `L.MLPConvolution2D` for implementing this method.
(originally, this is for 1x1 convolution in image recognition.
but we can use)

Q. Change `Model.__call__` to use `L.MLPConvolution2D` as the MLP part.


# Training

## Weight decay

*Weight decay* is a common method to regularize deep learning models.
In each parameter update, we shrink the parameters as follows:

```
w <- w - eta w
```
where `eta` is a hyper parameter that determines the amount of regularization.

It is equivalent to online L2 regularization.

In Chainer, we can realize weight decay as a hook to optimizers.



The hook is any callable that takes the hooked optimizer in its `__call__` method.

We apply weight decay, following the original paper [1].


```python
optimizer.add_hook(WeightDecay(1e-3))
```

## TestEvaluator

As the model includes dropout, which should behave differently in training and test phases, we need to change it accordingly.





# Reference
[1] Li, Z., & Yu, Y. (2016). Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. arXiv preprint arXiv:1604.07176.
