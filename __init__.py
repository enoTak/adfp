from pyautodiff.core_simple import Variable
from pyautodiff.core_simple import Function
from pyautodiff.core_simple import numerical_diff
from pyautodiff.functions import square, exp
from pyautodiff.utils import using_config, no_grad
from pyautodiff.arithmetic_operator import setup_variable
from pyautodiff.utils import get_dot_graph, plot_dot_graph


setup_variable()