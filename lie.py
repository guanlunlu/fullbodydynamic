import numpy as np


def skewMat(x):
    x1, x2, x3 = x
    return np.array([[0, -x3, x2], [x3, 0, -x1], [-x2, x1, 0]])


w1 = np.array([1, 2, 3])
w2 = np.array([4, 5, 6])
w12_cross = np.cross(w1, w2)
w12_cross_skew = skewMat(w12_cross)

lfs = skewMat(w1) @ skewMat(w2) - skewMat(w2) @ skewMat(w1)
print(w12_cross_skew)
print(lfs)
