from chainer import functions as F
from chainer import links as L
from chainer import reporter


class Classifier(L.Classifier):

    def __call__(self, *args):
        x = args[:-3]
        structure_label = args[-3]
        absolute_solvent_label = args[-2]
        relative_solvent_label = args[-1]

        (structure_pred, absolute_solvent_pred,
         relative_solvent_pred) = self.predictor(*x)

        structure_loss = F.softmax_cross_entropy(
            structure_pred, structure_label)
        absolute_solvent_loss = F.sigmoid_cross_entropy(
            absolute_solvent_pred, absolute_solvent_label)
        relative_solvent_loss = F.sigmoid_cross_entropy(
            relative_solvent_pred, relative_solvent_label)

        self.loss = (structure_loss +
                     absolute_solvent_loss + relative_solvent_loss)
        reporter.report({'structure_loss': structure_loss,
                         'absolute_solvent_loss': absolute_solvent_loss,
                         'relative_solvent_loss': relative_solvent_loss,
                         'loss': self.loss}, self)

        structure_accuracy = F.accuracy(structure_pred, structure_label)
        absolute_solvent_accuracy = F.binary_accuracy(
            absolute_solvent_pred, absolute_solvent_label)
        relative_solvent_accuracy = F.binary_accuracy(
            relative_solvent_pred, relative_solvent_label)
        reporter.report(
            {'accuracy': structure_accuracy,
             'absolute_solvent_accuracy': absolute_solvent_accuracy,
             'relative_solvent_accuracy': relative_solvent_accuracy},
            self)
        return self.loss
