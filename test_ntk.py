from computational_graphs.models.custom_modules.custom_nsga_net.old_evo_net import OldEvoNet

from data_loaders.cifar10 import Cifar10

from utils.neural_net.get_ntk import get_ntk_n
from utils.neural_net.kaiming_norm import init_model
from utils.neural_net.linear_region_counter import Linear_Region_Collector

import numpy as np

model = OldEvoNet(input_size=[3, 32, 32],
                output_size=10,
                n_bits={"kernel_sizes": 2, "pool_sizes": 1, "channels": 2},
                target_val= {
                    "kernel_sizes": [3, 5, 7, 9],
                    "pool_sizes": [1, 2],
                    "channels": [16, 32, 64, 128]
                },
                n_nodes=[6]*3,
                genome='0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 1 1 0 0 1 0 1 1 1 0 0 0 1 0 0 0 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 0 0 0 0 0 0')

data_loader = Cifar10(data_folder='./data/cifar10-python',
                    num_workers=4,
                    batch_size=32,
                    pin_memory=True,
                    cutout=True,
                    cutout_length=16,
                    drop_last=True)


train_loader = data_loader.train_loader

init_model(model)
model = model.cuda()
data_iter = iter(train_loader)
x_loader, _ = data_iter.next()
ntks, lrcs = [], []
for i in range(3):
    ntk = get_ntk_n(train_loader, [model], num_batch=1, train_mode=False)
    ntks += ntk

    lrc_model = Linear_Region_Collector(input_size=x_loader.size(), 
                                        sample_batch=3, 
                                        data_loader=train_loader, 
                                        seed=1,
                                        models=[model])

    lrc = lrc_model.forward_batch_sample()
    lrcs += lrc
    lrc_model.clear()

ntks = np.array(ntks)
lrcs = np.array(lrcs)

print(ntks.mean())
print(lrcs.mean())

