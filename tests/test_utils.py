import sys
sys.path.append("../.")


import unittest
import numpy as np
from pyautodiff import Variable
from pyautodiff.analytic_function import square
from pyautodiff.utils import no_grad


class NoGradContextTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
        self.assertFalse(x.is_updated_grad)