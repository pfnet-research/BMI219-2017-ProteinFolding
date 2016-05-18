from chainer import datasets
import numpy
import six


def load(fname, debug, V, C, T=700, L=57):
    a = numpy.load(fname)
    N = len(a)
    a = a.reshape(N, T, L)
    acids_raw = a[..., :V + 1]
    structure_labels_raw = a[..., V + 1:V + C + 2]
    absolute_solvent_labels = a[..., V + C + 4].astype(numpy.int32)
    relative_solvent_labels = a[..., V + C + 5].astype(numpy.int32)
    profiles = a[..., -V - 1:-1].astype(numpy.float32)

    acids = numpy.full((N, T), numpy.nan, dtype=numpy.int32)
    structure_labels = numpy.full((N, T), numpy.nan, dtype=numpy.int32)

    for i in six.moves.range(V):
        acids[acids_raw[..., i] == 1.0] = i
    acids[acids_raw[..., V] == 1.0] = -1
    assert not (numpy.isnan(acids)).any()
    for i in six.moves.range(C):
        structure_labels[structure_labels_raw[..., i] == 1.0] = i
    structure_labels[structure_labels_raw[..., C] == 1.0] = -1
    assert not (numpy.isnan(structure_labels)).any()

    if debug:
        acids = acids[..., :100]
        structure_labels = structure_labels[..., :100]
        profiles = profiles[:, :100, :]
        absolute_solvent_labels = absolute_solvent_labels[..., :100]
        relative_solvent_labels = relative_solvent_labels[..., :100]

    return datasets.TupleDataset(acids, profiles, structure_labels,
                                 absolute_solvent_labels,
                                 relative_solvent_labels)
