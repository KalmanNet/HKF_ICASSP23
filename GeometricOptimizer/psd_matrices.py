import torch

"""
Implementation of the Riemann space of PSD matrices using the log-cholesky metric:

Lin, Zhenhua. "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition."
SIAM Journal on Matrix Analysis and Applications 40.4 (2019): 1353-1370.
"""
# Class defining the Riemann-Manifold on lower-triangular matrices using the log-Cholesky metric
class L_Matrices:

    def __init__(self, dim: int):
        self.dim = dim

    def metric(self, manifold_point: torch.Tensor, tangent_a: torch.Tensor, tangent_b: torch.Tensor) -> torch.Tensor:
        """
        Return the metric of two tangent vectors at a given point on the manifold
        :manifold_point: Point on Manifold
        :tangent_a: First tangent vector
        :tangent_b: Second tangent vector
        """
        mul_tangent = tangent_a * tangent_b

        triangular_part = torch.tril(mul_tangent, diagonal=-1)

        diagonal_part = torch.diagonal(mul_tangent)

        inverse_point = torch.matrix_power(manifold_point, -2)

        diagonal_part *= torch.diagonal(inverse_point)

        metric = triangular_part.sum() + diagonal_part.sum()

        return metric

    def geodesic(self, point: torch.Tensor, gradient: torch.Tensor, step: float) -> torch.Tensor:
        """
        Calculate the geodesic along a tangent vector
        """
        # Get lower triag. matrix part
        point_triag = torch.tril(point, -1)
        grad_triag = torch.tril(gradient, -1)

        # Get diagonal part
        point_diag = torch.diagonal(point)
        grad_diag = torch.diagonal(gradient)

        # Calculate exponential map
        exp = torch.diag_embed(torch.exp(step * grad_diag / point_diag))

        # Geodesic step
        gamma = point_triag + step * grad_triag + torch.diag_embed(point_diag) @ exp

        return gamma

    def exponential_map(self, point: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """
        Exponential map for space of triag matrices. Same as geodesic if the gradient is scales with the step size.
        """
        point_triag = torch.tril(point, -1)
        direction_triag = torch.tril(gradient, -1)

        point_diag = torch.diagonal(point)
        direction_diag = torch.diagonal(gradient)

        exp = torch.exp(direction_diag / point_diag)

        return point_triag + direction_triag + torch.diag_embed(point_diag * exp)


# Class defining the Riemann-Manifold on psd matrices using the log-Cholesky metric
class PSD_Matrices:

    def __init__(self, dim: int):

        # Matrix dimension
        self.dim = dim

        # Lower triag. Riemann-Manifold
        self.Lmats = L_Matrices(self.dim)

    @torch.no_grad()
    def metric(self, point: torch.Tensor, tangent_a: torch.Tensor, tangent_b: torch.Tensor) -> torch.Tensor:

        point_chol = torch.linalg.cholesky(point)

        dp_a = self.differential(point_chol, tangent_a)
        dp_b = self.differential(point_chol, tangent_b)

        metric = self.Lmats.metric(point_chol, dp_a, dp_b)

        return metric

    @torch.no_grad()
    def differential(self, point_chol: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """
        Differential map in psd space
        """
        point_chol_inv = torch.inverse(point_chol)
        dp = point_chol_inv @ tangent @ point_chol_inv.T
        dp = point_chol @ (torch.tril(dp, -1) + 0.5 * torch.diag_embed(torch.diagonal(dp)))

        return dp

    @torch.no_grad()
    def geodesic(self, point: torch.Tensor, gradient: torch.Tensor, step: float) -> torch.Tensor:
        try:
            point_chol = torch.linalg.cholesky(point)

            grad_diff = self.differential(point_chol, gradient)

            cholesky_space_geodesic = self.Lmats.geodesic(point_chol, grad_diff, step)

            gamma = cholesky_space_geodesic @ cholesky_space_geodesic.T
        except torch.linalg.LinAlgError:
            gamma = point

        return 0.5 * (gamma + gamma.T)

    @torch.no_grad()
    def exponential_map(self, point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:

        point_chol = torch.linalg.cholesky(point)
        differential = self.differential(point_chol, direction)

        cholesky_space_exp_map = self.Lmats.exponential_map(point_chol, differential)

        return cholesky_space_exp_map @ cholesky_space_exp_map.T

    @torch.no_grad()
    def norm(self, point: torch.Tensor, tang: torch.Tensor) -> torch.Tensor:

        res = torch.sqrt(self.metric(point, tang, tang))
        return res

    @torch.no_grad()
    def step(self, point: torch.Tensor, direction: torch.Tensor, step_size: float):
        return self.geodesic(point, direction, step_size)

    @torch.no_grad()
    def euclidean_to_riemann_gradient(self, point: torch.Tensor, gradient: torch.Tensor, is_symmetric: bool = True) \
            -> torch.Tensor:
        """
        Return the riemann gradient at a point given a euclidean gradient
        point: current point in psd space
        gradient: gradient to convert
        is_symmetric: if the current gradient is already symmetric
        """
        if is_symmetric:
            return gradient
        else:
            raise NotImplementedError
