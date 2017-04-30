from chainer.training import extensions as E


class Evaluator(E.Evaluator):

    def evaluate(self):
        predictor = self.get_target('main').predictor
        train = predictor.train
        predictor.train = False
        ret = super(Evaluator, self).evaluate()
        predictor.train = train
        return ret
