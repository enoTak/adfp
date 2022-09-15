import unittest
import numpy as np
from adfp import Variable
import adfp.matrix_functions as F
from adfp.calc_utils import array_equal


class ReshapeTest(unittest.TestCase):
    def test_reshape_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        actual = y
        expected = Variable(np.array([1, 2, 3, 4, 5, 6]))
        self.assertTrue(array_equal(actual, expected))

    def test_reshape_grad(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))


    def test_reshape_of_instance_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.reshape((6,))
        actual = y
        expected = Variable(np.array([1, 2, 3, 4, 5, 6]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.reshape(6)
        actual = y
        expected = Variable(np.array([1, 2, 3, 4, 5, 6]))
        self.assertTrue(array_equal(actual, expected))

    def test_reshape_of_instance_grad(self):
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


class TransposeTest(unittest.TestCase):
    def test_transpose_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        actual = y
        expected = Variable(np.array([[1, 4],[2, 5], [3, 6]]))
        self.assertTrue(array_equal(actual, expected))

    def test_transpose_grad(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

    def test_transpose_of_instance_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()
        actual = y
        expected = Variable(np.array([[1, 4],[2, 5], [3, 6]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.T
        actual = y
        expected = Variable(np.array([[1, 4],[2, 5], [3, 6]]))
        self.assertTrue(array_equal(actual, expected))

    def test_transpose_of_instance_grad(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.T
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

    def test_transpose_with_axes_value(self):
        x = Variable(np.array([[[1, 2, 3], [4, 5, 6]]]))
        y = x.transpose(2, 0, 1)
        actual = y
        expected = Variable(np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))
        self.assertTrue(array_equal(actual, expected))

        y = x.transpose((2, 0, 1))
        actual = y
        expected = Variable(np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))
        self.assertTrue(array_equal(actual, expected))

        y = x.transpose([2, 0, 1])
        actual = y
        expected = Variable(np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))
        self.assertTrue(array_equal(actual, expected))

    def test_transpose_with_axes_grad(self):
        x = Variable(np.array([[[1, 2, 3], [4, 5, 6]]]))
        y = Variable(np.array([[[7, 8, 9], [10, 11, 12]]]))
        z = x * y
        z = z.transpose(2, 0, 1)
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertTrue(array_equal(actual, expected))

        x.cleargrad()
        z = x * y
        z = z.transpose((2, 0, 1))
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertTrue(array_equal(actual, expected))

        x.cleargrad()
        z = x * y
        z = z.transpose([2, 0, 1])
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertTrue(array_equal(actual, expected))

        x.cleargrad()
        z = x * y
        z = z.transpose()
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertTrue(array_equal(actual, expected))