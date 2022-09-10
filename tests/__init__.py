from pyautodiff.core_simple.variable import Variable
from pyautodiff.core_simple.function import Function
from pyautodiff.core_simple.function import numerical_diff
from pyautodiff.functions import square, exp
from pyautodiff.utils import using_config, no_grad
from pyautodiff.core_simple.arithmetic_operator import setup_variable
from pyautodiff.utils import get_dot_graph, plot_dot_graph


setup_variable()