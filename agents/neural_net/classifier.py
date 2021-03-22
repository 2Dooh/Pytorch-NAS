import agents.neural_net.nn_agent as base

class Classifier(base.NNAgent):
    def __init__(self, config):
        super().__init__(config)

    def _predict(self, outputs, **kwargs):
        _, pred = outputs.max(1)
        return {'pred': pred}