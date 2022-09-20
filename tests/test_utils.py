import unittest
import numpy as np
from adfp.core import Variable
from adfp.functions.analytic import square
from adfp.config import no_grad


class NoGradContextTest(unittest.TestCase):
    def test_no_grad(self):
        with no_grad():
            x = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
        self.assertFalse(x.is_updated_grad)