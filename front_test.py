import numpy as np

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_problem



problem = get_problem('zdt4')

import torch
res = torch.load('experiments/ZDT4/out/result.pth.tar')

plot = Scatter()
plot.add(res.F, color="red")
plot.show()