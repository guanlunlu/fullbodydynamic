import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
arrow_prop_dict = dict(
    mutation_scale=20, arrowstyle="-|>", color="k", shrinkA=0, shrinkB=0
)
a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict)
ax.add_artist(a)

plt.show()
