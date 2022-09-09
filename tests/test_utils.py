import sys
sys.path.append("../.")


import unittest
import numpy as np
from numeric_ad.core_simple import Variable
from numeric_ad.functions import square
from numeric_ad.utils import no_grad


class NoGradContextTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
            flg = True
        self.assertTrue(flg)