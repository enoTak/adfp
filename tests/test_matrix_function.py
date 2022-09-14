import unittest
import numpy as np
from adfp import Variable
import adfp.matrix_functions as F
from adfp.calc_utils import array_equal


class MatrixFunctionTest(unittest.TestCase):
    def test_reshape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

    def test_reshape_of_instance(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.reshape((6,))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.reshape(6)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))
