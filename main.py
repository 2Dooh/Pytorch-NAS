from agents import *
import json
import torch
from graphs.models import LinearRegression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

config = None
with open('./configs/salary.json') as  json_file:
    config = json.load(json_file)
agent_class = globals()[config['agent']]

agent = agent_class(config)
agent.run()

(X_test, y_test) = agent.data_loader.test_loader.dataset.tensors
y_pred = agent.model(X_test)

plt.scatter(X_test, y_test, marker='x', color='red', label='test points')
plt.plot(X_test, y_pred, label='predict')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--')
plt.show()

print('Debug')