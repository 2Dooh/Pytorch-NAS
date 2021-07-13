from utils.config import process_config

import agents as agents

import click


@click.command()
@click.option('--config', '-cfg', required=True)
@click.option('--console_log', default=True, type=bool)
def cli(config, console_log):
    config = process_config(config, console_log=console_log)
    agent_constructor = getattr(agents, config.agent)
    agent = agent_constructor(config)
    agent.solve()

if __name__ == '__main__':
    # cli(['--config', './configs/dynamic_mlp.json'])
    cli()