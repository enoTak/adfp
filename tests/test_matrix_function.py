from cmath import exp
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

    def test_scaler(self):
        x = Variable(np.array(2))
        actual = x.reshape(1,1)
        expected = Variable(np.array([[2]]))
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


class BroadcastToTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.broadcast_to(x, (2, 3))
        actual = y
        expected = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3]]))
        y = F.broadcast_to(x, (2, 3))
        actual = y
        expected = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
        self.assertTrue(array_equal(actual, expected))

    def test_grad(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.broadcast_to(x, (2, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([2, 2, 2]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3]]))
        y = F.broadcast_to(x, (2, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[2, 2, 2]]))
        self.assertTrue(array_equal(actual, expected))

    def test_shape_diff(self):
        x = Variable(np.array([1, 2, 3]))
        error_occured = False
        try:
            y = F.broadcast_to(x, (2, 4))
        except ValueError:
            error_occured = True
        finally:
            self.assertTrue(error_occured)
            

class SumToTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum_to(x, (1, 3))
        actual = y
        expected = Variable(np.array([[5, 7, 9]]))
        self.assertTrue(array_equal(actual, expected))

        y = F.sum_to(x, (2, 1))
        actual = y
        expected = Variable(np.array([[6], [15]]))
        self.assertTrue(array_equal(actual, expected))

    def test_grad(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum_to(x, (1, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

        x.cleargrad()
        y = F.sum_to(x, (2, 1))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

    def test_shape_diff(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum_to(x, (5, 3))
        actual = y
        expected = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum_to(x, (5, 1))
        actual = y
        expected = Variable(np.array([[6], [15]]))
        self.assertTrue(array_equal(actual, expected))


class SumTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x)
        actual = y
        expected = Variable(np.array(21))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = F.sum(x)
        actual = y
        expected = Variable(np.array(21))
        self.assertTrue(array_equal(actual, expected))

    def test_value_with_options(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x, axis=0)
        actual = y
        expected = Variable(np.array([5, 7, 9]))
        self.assertTrue(array_equal(actual, expected))

        y = F.sum(x, keepdims=True)
        actual = y
        expected = Variable(np.array([[21]]))
        self.assertTrue(array_equal(actual, expected))

    def test_grad_with_option(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.sum(x, axis=0)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1, 1, 1], [1, 1, 1]]))
        self.assertTrue(array_equal(actual, expected))

        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = F.sum(x, keepdims=True)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([1, 1, 1, 1, 1, 1]))
        self.assertTrue(array_equal(actual, expected))

    def test_sum_of_instance(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.sum()
        actual = y
        expected = Variable(np.array(21))
        self.assertTrue(array_equal(actual, expected))


class MatMulTest(unittest.TestCase):
    def test_value(self):
        X = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        Y = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        Z = F.matmul(X, Y)

        actual = Z
        expected = Variable(np.array([[22, 28], [49, 64]]))
        self.assertTrue(array_equal(actual, expected))

    def test_grad(self):
        X = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        Y = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        Z = F.matmul(X, Y)
        Z.backward()

        actual = X.grad
        expected = Variable(np.ones_like(Z.data).dot(Y.data.T))
        self.assertTrue(array_equal(actual, expected))

        actual = Y.grad
        expected = Variable(X.data.T.dot(np.ones_like(Z.data)))
        self.assertTrue(array_equal(actual, expected))

    def test_value_vector(self):
        x = Variable(np.array([1, 2, 3]))
        Y = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        z = F.matmul(x, Y)
        z.backward()

        actual = z
        expected = Variable(np.array([22, 28]))
        self.assertTrue(array_equal(actual, expected))

        actual = x.grad
        expected = Variable(np.array([3, 7, 11]))
        self.assertTrue(array_equal(actual, expected))


class InnerProdTest(unittest.TestCase):
    def test_value(self):
        v = Variable(np.array([1, 2, 3]))
        w = Variable(np.array([2, 3, 4]))
        y = F.inner_prod(v, w)
       
        actual = y
        expected = Variable(np.array(20))
        self.assertTrue(array_equal(actual, expected))

    def test_grad(self):
        v = Variable(np.array([1, 2, 3]))
        w = Variable(np.array([2, 3, 4]))
        y = F.inner_prod(v, w)
        y.backward()

        actual = v.grad
        expected = w
        self.assertTrue(array_equal(actual, expected))

        actual = w.grad
        expected = v
        self.assertTrue(array_equal(actual, expected))