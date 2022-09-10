import sys
sys.path.append("../.")


import unittest
import numpy as np
from pyautodiff.core_simple.variable import Variable
from pyautodiff.analytic_function import square
from pyautodiff.utils import no_grad


class NoGradContextTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
        self.assertTrue(x.grad is None)