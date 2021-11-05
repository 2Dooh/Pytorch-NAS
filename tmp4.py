import torch

import numpy as np

import matplotlib.pyplot as plt

from pymoo.util.nds.non_dominated_sorting import find_non_dominated

from scipy.interpolate import griddata

import scipy as sp
import scipy.interpolate

baseline = torch.load('data/baseline201_2obj.pth.tar')
r = np.random.choice(np.arange(30))
r = 25
print('Random number: {}'.format(r))
F_baseline = np.array(baseline[r][2][-1])

plt.plot(F_baseline[:, 0][np.argsort(F_baseline[:, 0])], F_baseline[:, 1][np.argsort(F_baseline[:, 0])], linewidth=4)
plt.scatter(F_baseline[:, 0], F_baseline[:, 1], color='red', marker='o', s=85)
plt.xlabel('FLOPs (Millions)', fontsize=15)
plt.ylabel('Validation Error (%)', fontsize=15)
plt.grid(linestyle='-')
plt.savefig('assets/2d_front.pdf', bbox_inches='tight', pad_inches=0.0)
plt.show()

# gf = torch.load('data/gf201_3obj.pth.tar')
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D


# r = np.random.choice(np.arange(29))
# r = 11
# print('Random number: {}'.format(r))
# # gf_init = np.array(gf[r][2][0])
# F_gf = np.array(gf[r][2][-1])

# J = find_non_dominated(F_gf, F_gf)
# F_gf_I = F_gf[J]

# fig = plt.figure()
# ax = Axes3D(fig)
# fig.add_axes(ax)



# x, y, z = F_gf[:, 0], np.e**F_gf[:, 1], -F_gf[:, 2]

# x_grid = np.linspace(x.min(), x.max(), 100)
# y_grid = np.linspace(y.min(), y.max(), 100)
# B1, B2 = np.meshgrid(x_grid, y_grid)
# # Z = np.zeros((x.size, z.size))

# # spline = sp.interpolate.Rbf(x, y, z,  smooth=5, episilon=5)
# Z = griddata((x, y), z, (B1, B2), method='linear', rescale=True)
# # Z = spline(B1,B2)

# # ax.plot_wireframe(B1, B2, Z)
# ax.plot_surface(B1, B2, Z,alpha=0.5, rstride = 10, cstride = 10)

# # I = find_non_dominated(gf_init, gf_init)
# # gf_init_I = gf_init[I]


# # ax.scatter3D(gf_init[:, 0], np.e**gf_init[:, 1], -gf_init[:, 2], marker=".",  color='blue')
# # ax.scatter3D(gf_init_I[:, 0], np.e**gf_init_I[:, 1], -gf_init_I[:, 2], marker=".", label="initial front", color='blue')
# ax.scatter3D(x, y, z, marker="o", color='red')
# # ax.plot_surface(xi, yi, -zi)
# # ax.scatter3D(F_gf_I[:, 0], np.e**F_gf_I[:, 1], -F_gf_I[:, 2], marker=".", label="final front", color='red')
# ax.set_xlabel('FLOPs (Millions)', fontsize=15)
# ax.set_zlabel(r'$R_\mathcal{N}$', fontsize=15)
# ax.set_ylabel(r'$\kappa_{\mathcal{N}}$', fontsize=15)
# ax.set_xlim([12.3, 148.7])
# ax.set_ylim([314, 3273])

# ax.view_init(10.875, -60.375)
# # plt.gca().set_axis_off()
# plt.subplots_adjust(
#     top = 0.1, 
#     bottom = 0, 
#     right = .1, 
#     left = 0,         
#     hspace = 0, 
#     wspace = 0
# )
# plt.margins(0,0,0)
# plt.savefig('assets/3d_front.pdf', bbox_inches='tight', pad_inches=0.0)
# # plt.legend(loc='best')
# plt.show()

# print('ax.azim {}'.format(ax.azim))
# print('ax.elev {}'.format(ax.elev))