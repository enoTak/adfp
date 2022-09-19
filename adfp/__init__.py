from adfp.core.variable import Variable
from adfp.core.arithmetic_operator import setup_variable   
from adfp.core.function import Function

from adfp.calc_utils import numerical_diff, allclose
from adfp.functions.analytic_functions import *
from adfp.functions.matrix_functions import *

from adfp.config import using_config, no_grad
from adfp.utils import get_dot_graph, plot_dot_graph


setup_variable()
__version__ = '0.0.2'