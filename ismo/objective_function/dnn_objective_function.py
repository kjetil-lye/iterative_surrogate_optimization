import ismo.gradients


class DNNObjectiveFunction(object):
    def __init__(self, models : list,
                 J: callable):
        self.models = models
        self.J = J
        self.gradient =ismo.gradients.ChainRuleModel(models, J)

    def __call__(self, x):
        if len(x.shape) != 2:
            x = x.reshape(1, x.shape[0])

        u = np.array([model.predict(x) for model in self.models])

        return self.J(u)

    def grad(self, x):
        return self.gradient(x)

