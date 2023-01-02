from Filters.KalmanSmoother import KalmanFilter
import torch

from GeometricOptimizer.problem import problem
from GeometricOptimizer.optimizer import Rieman_Manifold_Gradient_Descent
from GeometricOptimizer.psd_matrices import PSD_Matrices


@torch.no_grad()
def cost_fn(Q_opt: torch.Tensor, R, P, rho_mean):
    S = R + P + Q_opt

    S_inv = torch.linalg.inv(S)

    cost = 0.5 * torch.log(torch.linalg.det(S))

    cost = cost + 0.5 * torch.mm(rho_mean.T, torch.mm(S_inv, rho_mean))

    return cost.squeeze()


@torch.no_grad()
def gradient_fn(Q_opt: torch.Tensor, R, P, rho_mean):
    S = R + P + Q_opt

    S_inv = torch.linalg.inv(S)
    s_rho = S_inv @ rho_mean

    grad = S_inv - s_rho @ s_rho.T
    return grad


class InterHKF(KalmanFilter):

    def __init__(self, T: int, em_vars=('R', 'Q'), n_residuals=5):
        self.m = T
        self.n = T
        self.Q = torch.eye(T)
        self.R_history = None
        super(InterHKF, self).__init__(sys_model=None, em_vars=em_vars, n_residuals=n_residuals)

        self.psd_matrices = PSD_Matrices(self.m)

        self.problem = problem(self.psd_matrices, cost_fn, gradient_fn)

        # Initialize RMGD solver
        self.RMGD = Rieman_Manifold_Gradient_Descent(self.problem, max_its=2, step_contraction=0.3,
                                                     initial_step_size=6
                                                     )

    def predict(self, t: int) -> None:
        """
        Prediction step
        :param t: Time index
        :return
        """
        # Predict the 1-st moment of x
        self.Predicted_State_Mean = self.Filtered_State_Mean

        # Predict the 2-nd moment of x
        self.Predicted_State_Covariance = self.Filtered_State_Covariance + self.get_Q(t)

        # Predict the 1-st moment of y
        self.Predicted_Observation_Mean = self.Predicted_State_Mean
        # Predict the 2-nd moment y
        self.Predicted_Observation_Covariance = self.Predicted_State_Covariance + self.get_R(t)

    def kgain(self, t: int) -> None:
        """
        Kalman gain calculation
        :param t: Time index
        :return: None
        """
        # Compute Kalman Gain
        self.KG = torch.linalg.pinv(self.Predicted_Observation_Covariance)
        self.KG = torch.bmm(self.Predicted_State_Covariance, self.KG)

    def correct(self) -> None:
        """
        Correction step
        :return:
        """
        # Compute the 1-st posterior moment
        self.Filtered_State_Mean = self.Predicted_State_Mean + torch.bmm(self.KG, self.Predicted_Residual)

        # Compute the 2-nd posterior moments
        self.Filtered_State_Covariance = self.Predicted_State_Covariance
        self.Filtered_State_Covariance = torch.bmm(self.KG, self.Filtered_State_Covariance)
        self.Filtered_State_Covariance = self.Predicted_State_Covariance - self.Filtered_State_Covariance

        self.Filtered_Residual = self.Observation - self.Filtered_State_Mean

    def update_R(self, R: torch.Tensor) -> None:
        self.R_history[:, self.t] = R
        self.R = R

    def update_Q(self, Q: torch.Tensor) -> None:
        self.Q_history[:, self.t] = Q
        self.Q = Q

    def update_online(self, observations: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Single step filtering
        :param observations: Observation at current time index
        :return: None
        """

        self.predict(self.t)
        self.kgain(self.t)
        self.innovation(observations)
        self.correct()

        # Update Arrays
        self.Filtered_State_Means[:, self.t] = self.Filtered_State_Mean
        self.Filtered_State_Covariances[:, self.t] = self.Filtered_State_Covariance
        self.Filtered_Observation_Means[:, self.t] = self.Filtered_State_Mean
        self.Filtered_Residuals[:, self.t] = self.Filtered_Residual

        self.Kalman_Gains[:, self.t] = self.KG

        self.Predicted_State_Means[:, self.t] = self.Predicted_State_Mean
        self.Predicted_State_Covariances[:, self.t] = self.Predicted_State_Covariance
        self.Predicted_Observation_Means[:, self.t] = self.Predicted_Observation_Mean
        self.Predicted_Observation_Covariances[:, self.t] = self.Predicted_Observation_Covariance
        self.Predicted_Residuals[:, self.t] = self.Predicted_Residual

        self.t += 1

        return self.Filtered_State_Mean, self.Filtered_State_Covariance

    def init_online(self, T: int) -> None:
        self.R_history = torch.empty(1, T, self.n, self.n)
        self.Q_history = torch.empty(1, T, self.m, self.m)

        super(InterHKF, self).init_online(T)

    def log_likelihood(self) -> torch.Tensor:

        cov = self.Predicted_Observation_Covariance

        res = torch.einsum('bij,bjk,bkl->bil',
                           (self.Predicted_Residual.mT, torch.linalg.pinv(cov), self.Predicted_Residual))
        res += torch.log(torch.linalg.det(cov))
        return -res

    def ml_update_q(self, observation: torch.Tensor, tmp=0) -> None:
        """
        Online upate the estimate for the process noise, using ML estimation
        :param observation: Tensor of current observations
        :return: None
        """

        Q_old = self.get_Q(self.t)
        alpha = 0.8

        # Get current Values
        rho = (observation - self.Filtered_State_Mean)
        R = self.get_R(self.t)
        P = self.Filtered_State_Covariance
        Q_curr = self.get_Q(self.t)

        # Reshape
        R = R.reshape(self.n, self.n)
        P = P.reshape(self.n, self.n)
        rho = rho.reshape(self.n, 1)

        # Optimize
        Q = self.RMGD.search(Q_curr, (R, P, rho))

        # Update
        self.update_Q(alpha * Q + (1 - alpha) * Q_old)
