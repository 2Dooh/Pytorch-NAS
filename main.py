from utils.config import process_config

import agents

import time

import logging

import click


@click.command()
@click.option('--config', required=True)
@click.option('--console_log', default=True, type=bool)
def cli(config, console_log):
    config = process_config(config, console_log=console_log)
    agent_constructor = getattr(agents, config.agent)
    agent = agent_constructor(config)
    start = time.time()
    agent.solve()
    end = time.time()
    logging.info('exec time: {}'.format(end-start))

if __name__ == '__main__':
    # cli(['--config', './configs/NASBenchDict.json'])
    cli()