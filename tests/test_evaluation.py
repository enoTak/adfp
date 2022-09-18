import unittest
import numpy as np

from adfp.core import Variable
from adfp.model.evaluation import mean_square_error


class MeanSquareErrorTest(unittest.TestCase):
    def test_value(self):
        x0 = Variable(np.array([1., 2., 3., 4.]))
        x1 = Variable(np.array([2., 1., 3., 6.]))
        y = mean_square_error(x0, x1)

        actual = y
        expected = Variable(np.array(1.5))
        self.assertEqual(actual, expected)

    def test_grad(self):
        x0 = Variable(np.array([1., 2., 3., 4.]))
        x1 = Variable(np.array([2., 1., 3., 6.]))
        y = mean_square_error(x0, x1)
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
        y = mean_square_error(x0, x1)

        actual = y
        expected = Variable(np.array(3.))
        self.assertEqual(actual, expected)