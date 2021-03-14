from utils.config import process_config

from agents.classifier import Classifier

path = './configs/mnist.json'
config = process_config(path)
agent = Classifier(config)
agent.solve()