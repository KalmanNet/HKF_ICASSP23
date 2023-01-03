

class problem:

    def __init__(self, manifold, cost_fn, gradient_fn):

        self.manifold = manifold
        self.cost_fn = cost_fn
        self.gradient_fn = gradient_fn

