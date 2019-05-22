from ismo.gradients import TensorGradient
import numpy as np


class ChainRuleModel(object):

    def __init__(self,
                 models: list,
                 J: callable):

        self.J = J
        self.models = models

        self.model_grads = [TensorGradient(model) for model in models]

    def __call__(self, x):
        if len(x.shape) != 2:
            x = x.reshape(1, x.shape[0])

        u = np.array([model.predict(x) for model in self.models])

        grads_ml = [model_grad(x) for model_grad in self.model_grads]

        grad_J = self.J.grad(u)

        grad_full = np.zeros((x.shape[0]))

        for i in range(grad_full.shape[0]):
            for j in range(u.shape[0]):
                grad_full[i] += grad_J[j] * grads_ml[j][0, i]

        return grad_full.reshape(x.shape)
