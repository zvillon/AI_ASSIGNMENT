import numpy as np

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params_and_grads):
        for param, grad in params_and_grads:
            np.clip(grad, -5.0, 5.0, out=grad)
            param -= self.lr * grad