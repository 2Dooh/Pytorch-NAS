from .nn_agent import NNAgent

class Classifier(NNAgent):
    def __init__(self, config):
        super().__init__(config)

    def _predict(self, outputs, **kwargs):
        _, pred = outputs.max(1)
        return {'pred': pred}