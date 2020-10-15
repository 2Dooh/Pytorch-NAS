from agents import *
import json

config = None
with open('./configs/mnist.json') as  json_file:
    config = json.load(json_file)
agent_class = globals()[config['agent']]

agent = agent_class(**config)
agent.run()

# (X_test, y_test) = agent.valid_queue.dataset.tensors
# with torch.no_grad():
#     y_pred = agent.model(X_test)

# print(y_pred)
# print(y_test)

# score = r2_score(y_test, y_pred)
# print(score)