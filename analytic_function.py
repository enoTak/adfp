
import numpy as np
from pyautodiff.function import Function


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * -np.sin(x)
        return gx


def cos(x):
    return Cos()(x)