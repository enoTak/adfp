from adfp.model.parameter import Parameter


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name , value):
        if isinstance(value, Parameter):
            self._params.add(name)
        super().__setattr__(name,  value)