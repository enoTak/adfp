from pyautodiff.module_config import use_simple_core
if use_simple_core:
    from pyautodiff.core_simple.variable import Variable
    from pyautodiff.core_simple.arithmetic_operator import setup_variable   
else:
    from pyautodiff.core.variable import Variable
    from pyautodiff.core.arithmetic_operator import setup_variable   

from pyautodiff.function import Function
from pyautodiff.calc_utils import numerical_diff, allclose
from pyautodiff.analytic_function import square, exp, sin, cos, tanh
from pyautodiff.config import using_config, no_grad
from pyautodiff.utils import get_dot_graph, plot_dot_graph


setup_variable()
__version__ = '0.0.1'