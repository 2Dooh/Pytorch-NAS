from nats_bench import create

import torch

import numpy as np

import click

@click.command()
@click.option('--dataset', '-dts', default='cifar10')
@click.option('--hp', default='200')
@click.option('--search_space', '-ss', default='tss')
def cli(dataset, 
        hp, 
        search_space):
    api = create(None, search_space, fast_mode=True, verbose=False)
    F = []
    for i, arch_str in enumerate(api):
        cost_info = api.get_cost_info(i, dataset, hp=hp)
        more_info = api.get_more_info(i, dataset, hp=hp, is_random=False)
        F += [[cost_info, more_info]]
        print(i)

    F = np.array(F).reshape((len(api), -1))

    torch.save({'obj': F}, 'experiments/[{}-{}-{}].pth.tar'.format(dataset, search_space, hp))

if __name__ == '__main__':
    cli()
