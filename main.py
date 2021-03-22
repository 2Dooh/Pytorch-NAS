from utils.config import process_config

import agents.evo_alg.mo_eas as agents

import time

name = 'NSGAII'
path = './configs/NSGA_II.json'
config = process_config(path)
agent = getattr(agents, name)(config)
start = time.time()
agent.solve()
end = time.time()
print('exec time: {}'.format(end-start))