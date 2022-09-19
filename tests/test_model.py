import unittest
import numpy as np
from adfp.core import Variable
from adfp.model.parameter import Parameter
from adfp.model.layer import Layer


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