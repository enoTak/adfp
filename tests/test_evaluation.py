import unittest
import numpy as np

from adfp.core import Variable
import adfp.model.evaluation as F
from adfp.calc_utils import numerical_diff, allclose


class MeanSquareErrorTest(unittest.TestCase):
    def test_value(self):
        x0 = Variable(np.array([1., 2., 3., 4.]))
        x1 = Variable(np.array([2., 1., 3., 6.]))
        y = F.mean_square_error(x0, x1)

        actual = y
        expected = Variable(np.array(1.5))
        self.assertEqual(actual, expected)

    def test_grad(self):
        x0 = Variable(np.array([1., 2., 3., 4.]))
        x1 = Variable(np.array([2., 1., 3., 6.]))
        y = F.mean_square_error(x0, x1)
        y.backward()

        actual = x0.grad
        expected = Variable(np.array([-.5, .5, 0., -1.]))
        self.assertEqual(actual, expected)

        actual = x1.grad
        expected = Variable(np.array([.5, -.5, 0., 1.]))
        self.assertEqual(actual, expected)

    def test_matrix(self):
        # division number = len of 1st tensor, not num of total elements
        x0 = Variable(np.array([[1., 2., 3.], [3., 4., 5,]]))
        x1 = Variable(np.array([[2., 1., 3.], [3., 6., 5.]]))
        y = F.mean_square_error(x0, x1)

        actual = y
        expected = Variable(np.array(3.))
        self.assertEqual(actual, expected)


class SoftmaxTest(unittest.TestCase):
    def test_vector_value(self):
        v = np.array([1., 2., 3.])
        x = Variable(v)
        y = F.softmax(x)

        actual = y
        expected = Variable(softmax_test(v, axis=None))
        self.assertEqual(actual, expected)

    def test_vector_gradient_check(self):
        v = np.array([1., 2., 3.])
        x = Variable(v)
        y = F.softmax(x)
        y.backward()

        num_grad = numerical_diff(F.softmax, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_matrix_value(self):
        v = np.array([[1., 2., 3.], [4., 5., 6.]])
        x = Variable(v)
        y = F.softmax(x)

        actual = y
        expected = Variable(softmax_test(v, axis=1))
        self.assertEqual(actual, expected)

    def test_matrix_gradient_check(self):
        v = np.array([[1., 2., 3.], [4., 5., 6.]])
        x = Variable(v)
        y = F.softmax(x)
        y.backward()

        num_grad = numerical_diff(F.softmax, x)
        flg = allclose(x.grad, num_grad)
        self.assertTrue(flg)


def softmax_test(x, axis):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp