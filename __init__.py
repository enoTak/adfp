from autodiff.core_simple import Variable
from autodiff.core_simple import Function
from autodiff.core_simple import numerical_diff
from autodiff.functions import square, exp
from autodiff.utils import using_config, no_grad
from autodiff.arithmetic_operator import setup_variable
from autodiff.utils import get_dot_graph, plot_dot_graph


setup_variable()