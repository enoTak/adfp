import numpy as np
from adfp.config import using_config
import adfp


class Variable:
    #---- the setting to be prior to binary operation of numpy.array ----
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported in Variable.')

        self.data = data
        self.name = name
        self._grad_inner = None
        self.creator = None
        self.generation = 0

    def __eq__(self, other):
        if np.isscalar(self.data) and np.isscalar(other.data):
            return self.data == other.data 
        return np.array_equal(self.data, other.data)
        
    def __ne__(self, other):
        if np.isscalar(self.data) and np.isscalar(other.data):
            return self.data != other.data 
        return not np.array_equal(self.data, other.data)

    #---- utility functions ----#
    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def grad(self):
        if self._grad_inner is None:
            return Variable(np.zeros_like(self.data))
        else:
            return self._grad_inner

    @property
    def is_updated_grad(self):
        return self._grad_inner is not None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'variable({p})'

    #---- matrix calculation ----#
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return adfp.functions.matrix.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return adfp.functions.matrix.transpose(self, axes)

    @property
    def T(self):
        return adfp.functions.matrix.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return adfp.functions.matrix.sum(self, axis, keepdims)

    def dot(self, Y):
        return adfp.functions.matrix.dot(self, Y)

    #---- main functions for autodifferentials ----#
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self._grad_inner is None:
            self._grad_inner = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f is None:
                return
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                    
                for x, gx in zip(f.inputs, gxs):
                    if x._grad_inner is None:
                        x._grad_inner = gx
                    else:
                        x._grad_inner = x._grad_inner + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y()._grad_inner = None

    def cleargrad(self):
        self._grad_inner = None


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)