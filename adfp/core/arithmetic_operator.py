import adfp
from adfp.core.variable import Variable
from adfp.function import Function, as_array


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape  = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = adfp.functions.matrix_functions.sum_to(gx0, self.x0_shape)
            gx1 = adfp.functions.matrix_functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
        

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape  = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = adfp.functions.matrix_functions.sum_to(gx0, self.x0_shape)
            gx1 = adfp.functions.matrix_functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
        

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
        

class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape  = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy * x1, gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = adfp.functions.matrix_functions.sum_to(gx0, self.x0_shape)
            gx1 = adfp.functions.matrix_functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):

    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape  = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = adfp.functions.matrix_functions.sum_to(gx0, self.x0_shape)
            gx1 = adfp.functions.matrix_functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
        

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow