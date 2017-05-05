# Protein secondary structure prediction with cascaded CNN and RNN

This is an example of the application of deep learning to protein secondary structure prediction.
This example is based on [1], but some minor modifications are applied.

See commentary.md for a detailed explanation


# Dependency

* [Chainer](http://chainer.org)
* [NumPy](http://www.numpy.org)
* [six](https://pypi.python.org/pypi/six)

# Usage

Retrieve dataset

```
bash get_data.sh
```

Train
```
PYTHONPATH="." python tools/train.py
```

# Reference

[1] Li, Z., & Yu, Y. (2016). Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks. arXiv preprint arXiv:1604.07176.
