
from functools import partial, wraps

from collections import namedtuple

import re

import sys

from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='pdf',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename, view=True)



def convert_genotype_to_config(arch):
        base_string = 'NetworkSelectorDatasetInfo:darts:'
        config = {}

        for cell_type in ['normal', 'reduce']:
            cell = eval('arch.' + cell_type)

            start = 0
            n = 2
            for node_idx in range(4):
                end = start + n
                ops = cell[2 * node_idx: 2 * node_idx + 2]

                # get edge idx
                edges = {base_string + 'edge_' + cell_type + '_' + str(start + i): op for
                         op, i in ops}
                config.update(edges)

                if node_idx != 0:
                    # get node idx
                    input_nodes = sorted(list(map(lambda x: x[1], ops)))
                    input_nodes_idx = '_'.join([str(i) for i in input_nodes])
                    config.update({base_string + 'inputs_node_' + cell_type + '_' + str(node_idx + 2): input_nodes_idx})

                start = end
                n += 1
        return config

def convert_config_to_genotype(config):
    base_string = 'NetworkSelectorDatasetInfo:darts:'
    for key, val in config.items():
        if 'edge' in key:
            pass
        elif 'input' in key:
            pass

only_numeric_fn = lambda x: int(re.sub("[^0-9]", "", x))
custom_sorted = partial(sorted, key=only_numeric_fn)
def parse_config(config, cell_type):
        cell = []

        edges = custom_sorted(
            list(
                filter(
                    re.compile('.*edge_{}*.'.format(cell_type)).match,
                    config
                )
            )
        ).__iter__()
        nodes = custom_sorted(
            list(
                filter(
                    re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                    config
                )
            )
        ).__iter__()
        op_1 = config[next(edges)]
        op_2 = config[next(edges)]
        cell.extend([(op_1, 0), (op_2, 1)])
        for node in nodes:
            op_1 = config[next(edges)]
            op_2 = config[next(edges)]
            input_1, input_2 = map(int, config[node].split('_'))
            cell.extend([(op_1, input_1), (op_2, input_2)])
        return cell

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
genotype = Genotype(
    normal=[
        ('avg_pool_3x3', 0), #0
        ('dil_conv_3x3', 1), #0
        ('dil_conv_3x3', 0), #1
        ('dil_conv_5x5', 1), #1
        ('dil_conv_5x5', 1), #2
        ('dil_conv_3x3', 1), #2
        ('avg_pool_3x3', 0), #3
        ('skip_connect', 1)],#3 
    normal_concat=[2, 3, 4, 5], 
    reduce=[
        ('skip_connect', 0), 
        ('max_pool_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_5x5', 1), 
        ('skip_connect', 2), 
        ('max_pool_3x3', 2), 
        ('skip_connect', 1)], 
    reduce_concat=[3, 4, 5])

# config = convert_genotype_to_config(genotype)
# arch = parse_config(config, 'normal')
# print(config)
# plot(genotype.normal, 'gen_norm.pdf')
import numpy as np
def genotype_to_autodl_format(genotype):
    autodl_genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat connectN connects')

    cells = {'normal': None, 'reduce': None}
    for cell in ['normal', 'reduce']:
        lst = []
        blocks = eval('genotype.' + cell)
        for i in np.arange(len(blocks))[::2]:
            lst += [(blocks[i], blocks[i+1])]
        cells[cell] = lst

    new_genotype = autodl_genotype(normal=cells['normal'], 
                                    reduce=cells['reduce'], 
                                    normal_concat=genotype.normal_concat, 
                                    reduce_concat=genotype.reduce_concat,
                                    connectN=None,
                                    connects=None)

    print(new_genotype)
    return new_genotype

new_gen = genotype_to_autodl_format(genotype)