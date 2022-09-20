import unittest
import numpy as np
from adfp.core import Variable
import adfp.functions.matrix as F
from adfp.calc_utils import allclose
from adfp.utils import numerical_grad


class ReshapeTest(unittest.TestCase):
    def test_reshape_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.reshape(x, (6,))
        actual = y
        expected = Variable(np.array([1., 2., 3., 4., 5., 6.]))
        self.assertEqual(actual, expected)

    def test_reshape_grad(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.reshape(x, (6,))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

    def test_reshape_of_instance_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.reshape((6,))
        actual = y
        expected = Variable(np.array([1., 2., 3., 4., 5., 6.]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.reshape(6)
        actual = y
        expected = Variable(np.array([1., 2., 3., 4., 5., 6.]))
        self.assertEqual(actual, expected)

    def test_reshape_of_instance_grad(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.reshape((6,))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.reshape(6)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

    def test_scalar(self):
        x = Variable(np.array(2.))
        actual = x.reshape(1, 1)
        expected = Variable(np.array([[2.]]))
        self.assertEqual(actual, expected)


class TransposeTest(unittest.TestCase):
    def test_transpose_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.transpose(x)
        actual = y
        expected = Variable(np.array([[1., 4.],[2., 5.], [3., 6.]]))
        self.assertEqual(actual, expected)

    def test_transpose_grad(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.transpose(x)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

    def test_transpose_of_instance_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.transpose()
        actual = y
        expected = Variable(np.array([[1., 4.],[2., 5.], [3., 6.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.T
        actual = y
        expected = Variable(np.array([[1., 4.],[2., 5.], [3., 6.]]))
        self.assertEqual(actual, expected)

    def test_transpose_of_instance_grad(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.transpose()
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.T
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

    def test_transpose_with_axes_value(self):
        x = Variable(np.array([[[1., 2., 3.], [4., 5., 6.]]]))
        y = x.transpose(2, 0, 1)
        actual = y
        expected = Variable(np.array([[[1., 4.]], [[2., 5.]], [[3., 6.]]]))
        self.assertEqual(actual, expected)

        y = x.transpose((2, 0, 1))
        actual = y
        expected = Variable(np.array([[[1., 4.]], [[2., 5.]], [[3., 6.]]]))
        self.assertEqual(actual, expected)

        y = x.transpose([2, 0, 1])
        actual = y
        expected = Variable(np.array([[[1., 4.]], [[2., 5.]], [[3., 6.]]]))
        self.assertEqual(actual, expected)

    def test_transpose_with_axes_grad(self):
        x = Variable(np.array([[[1., 2., 3.], [4., 5., 6.]]]))
        y = Variable(np.array([[[7., 8., 9.], [10., 11., 12.]]]))
        z = x * y
        z = z.transpose(2, 0, 1)
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertEqual(actual, expected)

        x.cleargrad()
        z = x * y
        z = z.transpose((2, 0, 1))
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertEqual(actual, expected)

        x.cleargrad()
        z = x * y
        z = z.transpose([2, 0, 1])
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertEqual(actual, expected)

        x.cleargrad()
        z = x * y
        z = z.transpose()
        z.backward(retain_grad=True)
        actual = x.grad
        expected = y
        self.assertEqual(actual, expected)


class BroadcastToTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([1., 2., 3.]))
        y = F.broadcast_to(x, (2, 3))
        actual = y
        expected = Variable(np.array([[1., 2., 3.], [1., 2., 3.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.]]))
        y = F.broadcast_to(x, (2, 3))
        actual = y
        expected = Variable(np.array([[1., 2., 3.], [1., 2., 3.]]))
        self.assertEqual(actual, expected)

    def test_grad(self):
        x = Variable(np.array([1., 2., 3.]))
        y = F.broadcast_to(x, (2, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([2., 2., 2.]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.]]))
        y = F.broadcast_to(x, (2, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[2., 2., 2.]]))
        self.assertEqual(actual, expected)

    def test_shape_diff(self):
        x = Variable(np.array([1., 2., 3.]))
        error_occured = False
        try:
            y = F.broadcast_to(x, (2, 4))
        except ValueError:
            error_occured = True
        finally:
            self.assertTrue(error_occured)
            

class SumToTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum_to(x, (1, 3))
        actual = y
        expected = Variable(np.array([[5., 7., 9.]]))
        self.assertEqual(actual, expected)

        y = F.sum_to(x, (2, 1))
        actual = y
        expected = Variable(np.array([[6.], [15.]]))
        self.assertEqual(actual, expected)

    def test_grad(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum_to(x, (1, 3))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

        x.cleargrad()
        y = F.sum_to(x, (2, 1))
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

    def test_shape_diff(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum_to(x, (5, 3))
        actual = y
        expected = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum_to(x, (5, 1))
        actual = y
        expected = Variable(np.array([[6.], [15.]]))
        self.assertEqual(actual, expected)


class SumTest(unittest.TestCase):
    def test_value(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum(x)
        actual = y
        expected = Variable(np.array(21.))
        self.assertEqual(actual, expected)

        x = Variable(np.array([1., 2., 3., 4., 5., 6.]))
        y = F.sum(x)
        actual = y
        expected = Variable(np.array(21.))
        self.assertEqual(actual, expected)

    def test_value_with_options(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum(x, axis=0)
        actual = y
        expected = Variable(np.array([5., 7., 9.]))
        self.assertEqual(actual, expected)

        y = F.sum(x, keepdims=True)
        actual = y
        expected = Variable(np.array([[21.]]))
        self.assertEqual(actual, expected)

    def test_grad_with_option(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum(x, axis=0)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([[1., 1., 1.], [1., 1., 1.]]))
        self.assertEqual(actual, expected)

        x = Variable(np.array([1., 2., 3., 4., 5., 6.]))
        y = F.sum(x, keepdims=True)
        y.backward(retain_grad=True)
        actual = x.grad
        expected = Variable(np.array([1., 1., 1., 1., 1., 1.]))
        self.assertEqual(actual, expected)

    def test_sum_of_instance(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = x.sum()
        actual = y
        expected = Variable(np.array(21.))
        self.assertEqual(actual, expected)


class MatMulTest(unittest.TestCase):
    def test_value(self):
        X = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        Y = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        Z = F.matmul(X, Y)

        actual = Z
        expected = Variable(np.array([[22., 28.], [49., 64.]]))
        self.assertEqual(actual, expected)

    def test_grad(self):
        X = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        Y = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        Z = F.matmul(X, Y)
        Z.backward()

        actual = X.grad
        expected = Variable(np.ones_like(Z.data).dot(Y.data.T))
        self.assertEqual(actual, expected)

        actual = Y.grad
        expected = Variable(X.data.T.dot(np.ones_like(Z.data)))
        self.assertEqual(actual, expected)

    def test_vector_error(self):
        x = Variable(np.array([1., 2., 3.]))
        Y = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        error_occured = False
        try:
            z = F.matmul(x, Y)
        except ValueError:
            error_occured = True
        finally:
            self.assertTrue(error_occured)


class InnerProdTest(unittest.TestCase):
    def test_value(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        y = F.inner_prod(v, w)
       
        actual = y
        expected = Variable(np.array(20.))
        self.assertEqual(actual, expected)

    def test_grad(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        y = F.inner_prod(v, w)
        y.backward()

        actual = v.grad
        expected = w
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v
        self.assertEqual(actual, expected)


class DotTest(unittest.TestCase):
    def test_vector_vector(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        y = F.dot(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.array(np.dot(v.data, w.data)))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = w
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v
        self.assertEqual(actual, expected)

    def test_scalar_vector(self):
        v = Variable(np.array(2.))
        w = Variable(np.array([2., 3., 4.]))
        y = F.dot(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.dot(v.data, w.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = w.sum()
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.broadcast_to(v, w.shape)
        self.assertEqual(actual, expected)

    def test_vector_scalar(self):
        v = Variable(np.array([2., 3., 4.]))
        w = Variable(np.array(2.))
        y = F.dot(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.dot(v.data, w.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.broadcast_to(w, v.shape)
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v.sum()
        self.assertEqual(actual, expected)

    def test_matrix_vector(self):
        v = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        w = Variable(np.array([1., 2., 3.]))
        y = F.dot(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.dot(v.data, w.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.broadcast_to(w, v.shape)
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.sum_to(v, w.shape)
        self.assertEqual(actual, expected)

    def test_vector_matrix(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        y = F.dot(v, w)
        y.backward()

        actual = y
        expected = Variable(np.dot(v.data, w.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.sum_to(w.T, v.shape)
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.broadcast_to(v.reshape(3, 1), w.shape)
        self.assertEqual(actual, expected)

    def test_matrix_matrix(self):
        v = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        w = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        y = F.dot(v, w)
        y.backward()

        actual = y
        expected = Variable(np.dot(v.data, w.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.broadcast_to(F.sum_to(w.T, (1, v.shape[0])), v.shape) 
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.broadcast_to(F.sum_to(v.T, (w.shape[1], 1)), w.shape)
        self.assertEqual(actual, expected)


class TraceTest(unittest.TestCase):
    def test_value(self):
        X = Variable(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        y = F.trace(X)

        actual = y
        expected = Variable(np.array(15.))
        self.assertEqual(actual, expected)

    def test_grad(self):
        X = Variable(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        y = F.trace(X)
        y.backward()

        actual = X.grad
        expected = Variable(np.identity(3))
        self.assertEqual(actual, expected)

    def test_gradient_check(self):
        X = Variable(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]))
        y = F.trace(X)
        y.backward()
        actual = X.grad.data

        num_grad = numerical_grad(F.trace, X)
        expected = num_grad
        self.assertTrue(allclose(actual, expected))


class LinearTest(unittest.TestCase):
    def test_vector_vector(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        b = Variable(np.array(3))
        y = F.linear(v, w, b)
        y.backward()
       
        actual = y
        expected = Variable(np.array(np.dot(v.data, w.data) + b.data))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = w
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v
        self.assertEqual(actual, expected)

        actual = b.grad
        expected = Variable(np.array(1.))
        self.assertEqual(actual, expected)

    def test_vector_vector_no_bias(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        y = F.linear(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.array(np.dot(v.data, w.data)))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = w
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v
        self.assertEqual(actual, expected)

    def test_vector_vector_none_bias_instance(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([2., 3., 4.]))
        b = Variable(None)
        y = F.linear(v, w)
        y.backward()
       
        actual = y
        expected = Variable(np.array(np.dot(v.data, w.data)))
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = w
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = v
        self.assertEqual(actual, expected)

        self.assertFalse(b.is_updated_grad)

    def test_matrix_vector(self):
        v = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        w = Variable(np.array([7., 8., 9.]))
        b = Variable(np.array([3., 2.]))
        y = F.linear(v, w, b)
        y.backward()
       
        actual = y
        expected = Variable(np.dot(v.data, w.data) + b.data)
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.broadcast_to(w, v.shape)
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.sum_to(v, w.shape)
        self.assertEqual(actual, expected)

        actual = b.grad
        expected = Variable(np.ones_like(b.data))
        self.assertEqual(actual, expected)

    def test_vector_matrix(self):
        v = Variable(np.array([1., 2., 3.]))
        w = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        b = Variable(np.array([3., 2.]))
        y = F.linear(v, w, b)
        y.backward()

        actual = y
        expected = Variable(np.dot(v.data, w.data) + b.data)
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.sum_to(w, (len(v), 1)).reshape((len(v),))
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.broadcast_to(v.reshape(len(v), 1), w.shape)
        self.assertEqual(actual, expected)

        actual = b.grad
        expected = Variable(np.ones_like(b.data))
        self.assertEqual(actual, expected)

    def test_matrix_matrix(self):
        v = Variable(np.array([[1., 2., 3.], [2, 3, 4]]))
        w = Variable(np.array([[1., 2.], [3., 4.], [5., 6.]]))
        b = Variable(np.array([[3., 2.], [2., 1.]]))
        y = F.linear(v, w, b)
        y.backward()

        actual = y
        expected = Variable(np.dot(v.data, w.data) + b.data)
        self.assertEqual(actual, expected)

        actual = v.grad
        expected = F.broadcast_to(F.sum_to(w.T, (1, v.shape[0])), v.shape) 
        self.assertEqual(actual, expected)

        actual = w.grad
        expected = F.broadcast_to(F.sum_to(v.T, (w.shape[1], 1)), w.shape)
        self.assertEqual(actual, expected)

        actual = b.grad
        expected = Variable(np.ones_like(b.data))
        self.assertEqual(actual, expected)