import numpy as np
from adfp.model.layer import Layer
from adfp.model.parameter import Parameter


class TargetFunc(Layer):
    def __init__(self, target, *init_values):
        super().__init__()
        self.target = target
        self.values = []

        for i, v in enumerate(init_values):
            p = Parameter(np.array(v))
            setattr(self, 'arg' + str(i), p)
            self.values.append(p)

    def forward(self):
        y = self.target(*(self.values))
        return y