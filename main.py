from agents import *
import json
from types import SimpleNamespace

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

config = None
with open('./configs/loan_prediction.json') as  json_file:
    config = json.load(json_file, object_hook=lambda d : SimpleNamespace(**d))
agent_class = globals()[config.agent]

agent = agent_class(config)
agent.run()
agent.finalize()