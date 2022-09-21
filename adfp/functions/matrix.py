import numpy as np

from adfp.core.function import Function, as_variable
import adfp.calc_utils as utils


__all__ = ['reshape', 'transpose', 'sum', 
           'broadcast_to', 'sum_to', 'matmul', 'inner_prod',
           'trace', 'linear', 'dot',
           'get_item']


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
        Z = X.dot(Y)
        return Z

    def backward(self, gZ):
        X, Y = self.inputs
        gX = matmul(gZ, Y.T)
        gY = matmul(X.T, gZ)
        return gX, gY


def matmul(X, Y):
    # numpy magic
    if isvector(X) or isvector(Y):
        raise ValueError('shapes {} or {} not accepted in matmul. need to reshape to matrix forms'.format(X.shape, Y.shape))
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


class Trace(Function):
    def forward(self, X):
        self.dim = len(X)
        return np.trace(X)

    def backward(self, gy):
        # assumed that gy is scalar
        return gy * np.identity(self.dim)


def trace(X):
    return Trace()(X)


class Linear(Function):
    def forward(self, x, W, b):
        y = np.dot(x, W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        if not isvector(x) and isvector(W):
            W_shape = W.shape
            W = as_column_vector(W)
            gy = as_column_vector(gy)
            gx = dot(gy, W.T).reshape(x.shape)
            gW = dot(x.T, gy).reshape(W_shape)
            return gx, gW, gb
        if isvector(x) and not isvector(W):
            x_shape = x.shape
            x = as_row_vector(x)
            gy = as_row_vector(gy)
            gx = dot(gy, W.T).reshape(x_shape)
            gW = dot(x.T, gy).reshape(W.shape)
            return gx, gW, gb
        gx = dot(gy, W.T)
        gW = dot(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gy

    def backward(self, ggx):
        return get_item(ggx, self.sllices)


# =============================================================================
# Utility functions for matrix calculation
# =============================================================================

def dot(X, Y):
    if isscalar(X) or isscalar(Y):
        return X * Y # applied broadcasting
    elif isvector(X) and isvector(Y):
        return inner_prod(X, Y)
    elif isvector(Y):
        Y = as_column_vector(Y)
        v = matmul(X, Y)
        return v.reshape(v.shape[0])
    elif isvector(X):
        X = as_row_vector(X)
        v = matmul(X, Y)
        return v.reshape(v.shape[1])
    return matmul(X, Y)


def isvector(x):
    return len(x.shape) == 1


def isscalar(x):
    return len(x.shape) == 0


def as_matrix(v):
    if isscalar(v):
        return v.reshape(1, 1)
    elif isvector(v):
        return v.reshape(1, len(v))
    else:
        return as_variable(v)


def as_column_vector(v):
    if isvector(v):
        return v.reshape(len(v), 1)
    else:
        return as_variable(v)


def as_row_vector(v):
    if isvector(v):
        return v.reshape(1, len(v))
    else:
        return as_variable(v)