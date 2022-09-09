import sys
sys.path.append("../.")


import unittest
import numpy as np
from autodiff.core_simple import Variable
from autodiff.functions import square
from autodiff.utils import no_grad


class NoGradContextTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
        self.assertTrue(x.grad is None)