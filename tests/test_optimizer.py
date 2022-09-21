import unittest

import adfp.model.optimizers as O
from adfp.model.target_func import TargetFunc


class OptimizerTest(unittest.TestCase):
    def setUp(self):
        self.lr = 0.05
        self.iters = 1000
        self.tor = 1e-5
        self.thres = 1e-4

    def test_sgd(self):
        opt = O.SGD(self.lr)
        x0, x1 = tester(opt, self.iters, self.tor)
        is_near = (x0.data < self.thres) and (x1.data < self.thres)
        self.assertTrue(is_near)

    def test_momentum(self):
        opt = O.MomentumSGD(self.lr)
        x0, x1 = tester(opt, self.iters, self.tor)
        is_near = (x0.data < self.thres) and (x1.data < self.thres)
        self.assertTrue(is_near)

    def test_adagrad(self):
        opt = O.AdaGrad(self.lr)
        x0, x1 = tester(opt, self.iters, self.tor)
        is_near = (x0.data < self.thres) and (x1.data < self.thres)
        self.assertTrue(is_near)


def tester(opt, iters, tor):
    init_x0 = 1.0
    init_x1 = 1.0
    target = TargetFunc(quad, init_x0, init_x1)
    opt.setup(target)
    for i in range(iters):
        y = target()
        target.cleargrads()
        y.backward()
        opt.update()

        x0 = target.arg0
        x1 = target.arg1

        if (abs(x0.grad.data) + abs(x1.grad.data)) < tor:
            break

    return x0, x1


def quad(x0, x1):
    y = x0 ** 2 + x1 ** 2
    return y