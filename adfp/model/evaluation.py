import numpy as np
from adfp.core import Function
import adfp.functions as F

class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = F.broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_square_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Softmax(Function):
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x):
        c = np.max(x)
        y = np.exp(x - c)
        sum_y = np.sum(y, axis=self.axis, keepdims=True)
        return y / sum_y

    def backward(self, gy):
        y, = self.outputs
        y = y()
        z = F.sum(y * gy, axis=self.axis, keepdims=True)
        gx = y * (-gy + z)
        return gx


def softmax(x):
    if x.ndim == 1:
        axis = None
    else:
        axis = 1
    return Softmax(axis)(x)