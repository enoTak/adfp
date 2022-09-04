from numeric_ad.core_simple import Variable, Function


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)


def add(x0, x1):
    return Add()(x0, x1)