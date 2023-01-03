import torch


class Rieman_Manifold_Gradient_Descent:

    def __init__(self,
                 problem,
                 max_its: int = 10,
                 initial_step_size: float = 1.,
                 sufficient_decrease: float = 1e-4,
                 step_contraction: float = 0.5):

        self.manifold = problem.manifold

        self.cost_fn = problem.cost_fn

        self.gradient_fn = problem.gradient_fn

        self.max_its = max_its

        self.initial_step_size = initial_step_size

        self.sufficient_decrease = sufficient_decrease

        self.step_contraction = step_contraction

        self.oldf = None

        self.optimism = 2

    @torch.no_grad()
    def search(self, x0, additional_input: tuple = ()):

        try:
            current_f = self.cost_fn(x0, *additional_input)
            current_grad = self.gradient_fn(x0, *additional_input)
            riemann_grad = self.manifold.euclidean_to_riemann_gradient(x0, current_grad)

            grad_norm = self.manifold.norm(x0, riemann_grad)

            if self.oldf is not None:
                alpha = 2 * (current_f - self.oldf) / (-(grad_norm ** 2))
                alpha *= self.optimism
            else:
                alpha = self.initial_step_size / grad_norm

            x_new = self.manifold.step(x0, -riemann_grad, alpha)
            f_new = self.cost_fn(x_new, *additional_input)

            num_its = 1

            while f_new > current_f + -(grad_norm ** 2) * self.sufficient_decrease * alpha and num_its <= self.max_its:
                alpha = self.step_contraction * alpha

                x_new = self.manifold.step(x0, -riemann_grad, alpha)
                f_new = self.cost_fn(x0, *additional_input)

                num_its += 1

            if f_new > current_f:
                x_new = x0

            self.oldf = current_f

        except torch.linalg.LinAlgError:
            x_new = x0

        return x_new
