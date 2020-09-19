
class Agent:
    def __init__(self, config):
        self.config = config

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def save_checkpoint(self, filename=None):
        # if filename is None:
        #     filename = datetime.datetime.now().strftime('%Y%m%d-%H%M') + 'pth.tar'
        
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def predict(self, sample):
        raise NotImplementedError