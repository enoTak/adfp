import sys
sys.path.append("..")


import unittest
import numpy as np
from numeric_ad.binary_operators import *


class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        actual = y.data
        expected = np.array(5.0)
        self.assertEqual(actual, expected)