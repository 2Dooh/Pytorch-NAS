problem = []

import numpy as np

n_params = 18
k = 3
indices = np.arange(n_params)
for i in indices[::k]:
    problem += [indices[i:i+k]]

print(problem)