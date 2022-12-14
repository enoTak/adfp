from adfp.module_config import use_simple_core
if use_simple_core:
    from adfp.core_simple.variable import Variable
    from adfp.core_simple.arithmetic_operator import setup_variable   
else:
    from adfp.core.variable import Variable
    from adfp.core.arithmetic_operator import setup_variable   

from adfp.core.function import Function
from adfp.calc_utils import numerical_diff, allclose, array_equal
from adfp.functions.analytic_functions import *
from adfp.functions.matrix_functions import *

from adfp.config import using_config, no_grad
from adfp.utils import get_dot_graph, plot_dot_graph


setup_variable()
__version__ = '0.0.2'