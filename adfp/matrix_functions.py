import numpy as np

from adfp.function import Function, as_variable
import adfp.calc_utils as utils


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes):
        self.axes = axes

    def forward(self, x):
        y = np.transpose(x, axes=self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        gx = transpose(gy, inv_axes)
        return gx


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy,
            self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, X, Y):
        self.X_shape = X.shape
        Z = X.dot(Y)
        return Z

    def backward(self, gZ):
        X, Y = self.inputs
        X = as_matrix(X)
        Y = as_matrix(Y)
        gZ = as_matrix(gZ)
        gX = matmul(gZ, Y.T).reshape(self.X_shape)
        gY = matmul(X.T, gZ)
        return gX, gY


def matmul(X, Y):
    return MatMul()(X, Y)


class InnerProd(Function):
    def forward(self, v, w):
        return v.dot(w)

    def backward(self, gy):
        # assumed that gy is scalar
        v, w = self.inputs
        gv, gw = w * gy, v * gy
        return gv, gw


def inner_prod(v, w):
    return InnerProd()(v, w)


def as_matrix(v):
    if len(v.shape) == 0:
        return v.reshape(1, 1)
    elif len(v.shape) == 1:
        return v.reshape(1, len(v))
    else:
        return v
    