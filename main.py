from utils.config import process_config

import agents.ea_agent as agents

import time

import logging

name = 'NSGAII'
path = './configs/NASBenchDict.json'
config = process_config(path, console_log=False)
agent = agents.EvoAgent(config)
start = time.time()
agent.solve()
end = time.time()
logging.info('exec time: {}'.format(end-start))