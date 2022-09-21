import unittest
import numpy as np

from adfp.core import Variable
import adfp.functions as F
from adfp.model.parameter import Parameter
from adfp.model.layer import Layer
from adfp.model.layers.linear import Linear
from adfp.model.target_func import TargetFunc


class ParameterTest(unittest.TestCase):
    def test_instance(self):
        x = Variable(np.array(1.0))
        p = Parameter(np.array(1.0))
        y = x * p

        self.assertTrue(isinstance(p, Parameter))
        self.assertFalse(isinstance(x, Parameter))
        self.assertFalse(isinstance(y, Parameter))
        self.assertTrue(isinstance(y, Variable))


class LayerTest(unittest.TestCase):
    def test_init(self):
        layer = Layer()

        p1 = Parameter(np.array(1.0))
        layer.p1 = p1
        p2 = Parameter(np.array(1.0))
        layer.p2 = p2
        p3 = Variable(np.array(1.0))
        layer.p3 = p3
        p4 = 'test'
        layer.p4 = p4

        actual = layer._params
        expected = set(['p1', 'p2'])
        self.assertEqual(actual, expected)

    def test_multi_layers(self):
        l = Layer()

        p1 = Parameter(np.array(1.0))
        p2 = Parameter(np.array(2.0))
        l.p1 = p1
        l.p2 = p2
        
        ml = Layer()
        ml.l = l
        p = Parameter(np.array(3.0))
        ml.p = p

        actual = [p for p in ml.params]
        self.assertEqual(len(actual), 3)


class LinearLayerTest(unittest.TestCase):
    def test_forward(self):
        O = 4
        l = Linear(O)
        I = 3
        x = np.random.rand(I, 1)
        y = l(x)
        y.backward()

        actual_val = y
        actual_grad_W = l.W.grad
        actual_grad_b = l.b.grad
        
        # check the access to same W and b
        z = F.dot(x, l.W) + l.b
        z.backward()
        expected_val = z
        expected_grad_W = 0.5 * l.W.grad
        expected_grad_b = 0.5 * l.b.grad
        
        self.assertEqual(actual_val, expected_val)
        self.assertEqual(actual_grad_W, expected_grad_W)
        self.assertEqual(actual_grad_b, expected_grad_b)


class TargetFuncTest(unittest.TestCase):
    def test_call(self):
        init_x0 = 1.0
        init_x1 = 2.0
        
        def quad(x0, x1):
            return 2 * x0 ** 2 + x1 ** 2

        target = TargetFunc(quad, init_x0, init_x1)
        y = target()
        target.cleargrads()
        y.backward()
        x0 = target.arg0
        x1 = target.arg1

        actual = x0
        expected = Variable(np.array(1.0))
        self.assertEqual(actual, expected)

        actual = x1
        expected = Variable(np.array(2.0))
        self.assertEqual(actual, expected)

        actual = y
        expected = Variable(np.array(6.0))
        self.assertEqual(actual, expected)
    
        actual = x0.grad
        expected = Variable(np.array(4.0))
        self.assertEqual(actual, expected)

        actual = x1.grad
        expected = Variable(np.array(4.0))
        self.assertEqual(actual, expected)