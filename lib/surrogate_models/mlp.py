# from torch.nn import Module, Linear, ModuleList, ModuleDict, Sequential
# import torch.nn as nn

# import torch.nn.functional as F

# import numpy as np

# class MultiLayerPerceptron(Module):
#     def __init__(self, 
#                  layers_dim, 
#                  activations, 
#                  **kwargs):
#         super(MultiLayerPerceptron, self).__init__()

#         self.linears = ModuleList(
#             [Linear(in_features, out_features) \
#                 for in_features, out_features in zip(layers_dim, layers_dim[1:])]
#         )

#         # if '' not in activations:
#         #     self.activations = ModuleDict([
#         #         [key, func] for key, func in zip(
#         #             activations, 
#         #             map(lambda f: getattr(nn, f, None)(), activations)
#         #         )
#         #     ])
#         self.activations = ModuleList(
#             [getattr(nn, act, nn.Identity)() for act in activations]
#         )

#         assert(len(self.linears) == len(self.activations))

#         # else:
#         #     self.activations = {'': None}
#         # self.list_activations = activations

#     def forward(self, x):

#         for linear, f in zip(self.linears, self.activations):
#             x = linear(x)
#             x = f(x)

#         return x