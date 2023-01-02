import torch
from tqdm import trange
from utils.MovingAverage import moving_average
from SystemModels.BaseSysmodel import BaseSystemModel
from numpy import pi

class KalmanFilter:

    def __init__(self, sys_model: BaseSystemModel or None, em_vars: tuple = ('R', 'Q'), n_residuals: int = 1):

        self.ssModel = sys_model

        if sys_model is not None:
            self.m = sys_model.m
            self.n = sys_model.n

            self.f = sys_model.f
            self.h = sys_model.h

            self.Q = sys_model.Q
            self.R = sys_model.R

        self.Q_arr = None
        self.R_arr = None

        self.AllVars = ('R', 'Q', 'F', 'H', 'Sigma')

        self.em_vars = em_vars if not em_vars == 'all' else self.AllVars + ('Mu',)

        self.nResiduals = n_residuals

        self.F_em = False
        self.H_em = False

        self.init_sequence()


    def f_batch(self, states: torch.Tensor, t: int) -> torch.Tensor:
        """
        Evaluate the state evolution function for a batch of states
        :param states: A batch of states of size (batch_size, m, 1)
        :param t: Time index
        :return: A batch of evolved states
        """
        if self.F_em:
            predictions = torch.bmm(self.F_arr[t], states)
        else:
            predictions = torch.stack([self.f(state, t) for state in states])

        return predictions

    def h_batch(self, states: torch.Tensor, t: int) -> torch.Tensor:
        """
        Evaluate the state observation function for a batch of states
        :param states: A batch of states of size (batch_size, m, 1)
        :param t: Time index
        :return: A batch of observed states
        """
        if self.H_em:
            observations = torch.bmm(self.H_arr[t], states)
        else:
            observations = torch.stack([self.h(state, t) for state in states])

        return observations

    def update_Q(self, Q: torch.Tensor) -> None:
        """
        Update Q
        :param Q: Process covariance
        :return: None
        """
        self.Q = Q
        self.ssModel.update_Q(Q)

    def get_Q(self, t: int) -> torch.Tensor:
        """
        Returns the current values of Q from the Q_array if available, or else from Q itself
        :param t: Time values for Q_array
        :return: Current process covariance
        """
        if self.Q_arr == None:
            return self.Q
        else:
            return self.Q_arr[:, t]

    def update_R(self, R: torch.Tensor) -> None:
        """
        Update R
        :param R: Observation covariance
        :return: None
        """
        self.R = R
        self.ssModel.update_R(R)

    def get_R(self, t: int):
        """
        Returns the current values of R from the R_array if available, or else from R itself
        :param t: Time values for R_array
        :return: Current observation covariance
        """
        if self.R_arr == None:
            return self.R
        else:
            return self.R_arr[:, t]

    def reset_QR_array(self) -> None:
        """
        Reset Q and R arrays
        :return:
        """
        self.Q_arr = None
        self.R_arr = None

    def update_sysmodel(self, sys_model: BaseSystemModel) -> None:

        self.m = sys_model.m
        self.n = sys_model.n

        self.f = sys_model.f
        self.h = sys_model.h

    def init_sequence(self, initial_mean: torch.Tensor = None, initial_covariance: torch.Tensor = None) -> None:
        """
        Initialize both mean and covariance for the prior, default x_{0|0} ~ N(0,I)
        :param initial_mean: Mean of the initial distribution
        :param initial_covariance: Covariance of the initial distribution
        :return: None
        """
        # Initialize the mean
        self.init_mean(initial_mean)
        # Initialize the covariance
        self.init_covariance(initial_covariance)

    def init_mean(self, initial_mean: torch.Tensor = None) -> None:
        """
        Initialize the mean of the prior distribution
        :param initial_mean: Mean of the initial distribution
        :return: None
        """
        # Default
        if initial_mean == None:
            self.Initial_State_Mean = torch.zeros(self.m, 1)
        else:
            self.Initial_State_Mean = initial_mean

        # Add batch dimension if necessary
        if len(self.Initial_State_Mean.shape) == 2:
            self.Initial_State_Mean = self.Initial_State_Mean.unsqueeze(0)

    def init_covariance(self, initial_covariance: torch.Tensor = None) -> None:
        """
        Initialize the covariance of the prior distribution
        :param initial_covariance: Covariance of the initial distribution
        :return: None
        """
        # Default
        if initial_covariance is None:
            self.Initial_State_Covariance = torch.eye(self.m)
        else:
            self.Initial_State_Covariance = initial_covariance

        # Add batch dimension if necessary
        if len(self.Initial_State_Covariance.shape) == 2:
            self.Initial_State_Covariance = self.Initial_State_Covariance.unsqueeze(0)

        self.Initial_Observation_Covariance = self.Initial_State_Covariance

    def update_jacobians(self, t: int) -> None:
        """
        Update gradients for filtering process
        :param t: Time index
        :return: None
        """
        # Update Gradients
        if self.F_em:
            self.F = self.F_arr[t]
        else:
            self.F = torch.stack([self.ssModel.get_f_jacobian(state, t) for state in self.Filtered_State_Mean])

        if self.H_em:
            self.H = self.H_arr[t]
        else:
            self.H = torch.stack([self.ssModel.get_h_jacobian(state, t) for state in self.Filtered_State_Mean])

    def predict(self, t: int):
        """
        Prediction step
        :param t: Time index
        :return: None
        """

        # Predict the 1-st moment of x
        self.Predicted_State_Mean = self.f_batch(self.Filtered_State_Mean, t)

        # Compute Jacobians
        self.update_jacobians(t)

        # Predict the 2-nd moment of x
        self.Predicted_State_Covariance = torch.bmm(self.Filtered_State_Covariance, self.F.mT)
        self.Predicted_State_Covariance = torch.bmm(self.F, self.Predicted_State_Covariance)

        # Predict the 1-st moment of y
        self.Predicted_Observation_Mean = self.h_batch(self.Predicted_State_Mean, t)
        # Predict the 2-nd moment y
        self.Predicted_Observation_Covariance = torch.bmm(self.Predicted_State_Covariance, self.H.mT)
        self.Predicted_Observation_Covariance = torch.bmm(self.H, self.Predicted_Observation_Covariance)
        self.Predicted_Observation_Covariance += self.get_R(t)

    def kgain(self, t: int) -> None:
        """
        Kalman gain calculation
        :param t: Time index
        :return: None
        """
        # Compute Kalman Gain
        self.KG = torch.linalg.pinv(self.Predicted_Observation_Covariance)
        self.KG = torch.bmm(self.H.mT, self.KG)
        self.KG = torch.bmm(self.Predicted_State_Covariance, self.KG)

    def innovation(self, y: torch.Tensor) -> None:
        """
        Innovation step
        :param y: Observation of a state
        :return: None
        """
        self.Observation = y

        # Compute Innovation
        self.Predicted_Residual = (self.Observation - self.Predicted_Observation_Mean)

    def correct(self) -> None:
        """
        Correction step
        :return:
        """
        # Compute the 1-st posterior moment
        self.Filtered_State_Mean = self.Predicted_State_Mean + torch.bmm(self.KG, self.Predicted_Residual)

        # Compute the 2-nd posterior moments
        self.Filtered_State_Covariance = torch.bmm(self.H, self.Predicted_State_Covariance)
        self.Filtered_State_Covariance = torch.bmm(self.KG, self.Filtered_State_Covariance)
        self.Filtered_State_Covariance = self.Predicted_State_Covariance - self.Filtered_State_Covariance

        self.Filtered_State_Covariance  = 0.5 * self.Filtered_State_Covariance + 0.5 * self.Filtered_State_Covariance.mT

        self.Filtered_Residual = self.Observation - torch.bmm(self.H, self.Filtered_State_Mean)

    @torch.no_grad()
    def filter(self, observations: torch.Tensor, T: int) -> None:
        """
        Apply kalman filtering to the given observations
        :param observations: Tensor of observations
        :param T: Time horizon
        :return: None
        """

        # Add a batch dimension if there is none
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(0)
        observations = self.Observations = observations.unsqueeze(-1)

        # Compute Batch size
        self.BatchSize = observations.shape[0]

        if self.Initial_State_Mean.shape[0] == 1 and self.BatchSize != 1:
            self.Initial_State_Mean = self.Initial_State_Mean.repeat(self.BatchSize, 1, 1)

        if self.Initial_State_Covariance.shape[0] == 1 and self.BatchSize != 1:
            self.Initial_State_Covariance = self.Initial_State_Covariance.repeat(self.BatchSize, 1, 1)

        # Initialize sequences
        self.Filtered_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Filtered_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.Filtered_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
        self.Filtered_Residuals = torch.empty((self.BatchSize, T, self.n, 1))

        self.Kalman_Gains = torch.empty((self.BatchSize, T, self.m, self.n))

        self.Predicted_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Predicted_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.Predicted_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
        self.Predicted_Observation_Covariances = torch.empty((self.BatchSize, T, self.n, self.n))
        self.Predicted_Residuals = torch.empty((self.BatchSize, T, self.n, 1))

        self.F_arr = torch.empty((self.BatchSize, T, self.m, self.m))
        self.H_arr = torch.empty((self.BatchSize, T, self.n, self.m))

        # Initialize Parameters
        self.Filtered_State_Mean = self.Initial_State_Mean
        self.Filtered_State_Covariance = self.Initial_State_Covariance

        for t in range(T):
            self.predict(t)

            self.kgain(t)
            self.innovation(observations[:, t])
            self.correct()

            # Update Arrays
            self.Filtered_State_Means[:, t] = self.Filtered_State_Mean
            self.Filtered_Observation_Means[:, t] = self.h_batch(self.Filtered_State_Mean, t)
            self.Filtered_State_Covariances[:, t] = self.Filtered_State_Covariance
            self.Filtered_Residuals[:, t] = self.Filtered_Residual

            self.Kalman_Gains[:, t] = self.KG

            self.Predicted_State_Means[:, t] = self.Predicted_State_Mean
            self.Predicted_State_Covariances[:, t] = self.Predicted_State_Covariance
            self.Predicted_Observation_Means[:, t] = self.Predicted_Observation_Mean
            self.Predicted_Observation_Covariances[:, t] = self.Predicted_Observation_Covariance
            self.Predicted_Residuals[:, t] = self.Predicted_Residual

            self.F_arr[:, t] = self.F
            self.H_arr[:, t] = self.H

    def log_likelihood(self) -> torch.Tensor:

        res = torch.einsum('Tij,Tjk,Tkl->Til',
                            (self.Predicted_Residuals[0].mT,
                             torch.linalg.pinv(self.Predicted_Observation_Covariances[0]), self.Predicted_Residuals[0]))
        res += torch.log(torch.linalg.det(self.Predicted_Observation_Covariances[0])).reshape(-1,1,1)
        res += self.n * torch.log(2*torch.tensor(pi))
        return -0.5*res



    def init_online(self, T: int) -> None:
        """
        Initiliaze all data buffers for online filtering
        :param T: Time horizon
        :return: None
        """
        # Set batch size to 1
        self.BatchSize = 1

        if 'Initial_State_Mean' not in self.__dict__ or 'Initial_State_Covariance' not in self.__dict__:
            self.init_sequence()

        self.Filtered_State_Mean = self.Initial_State_Mean
        self.Filtered_State_Covariance = self.Initial_State_Covariance
        self.Predicted_Observation_Covariance = self.Initial_Observation_Covariance

        # Initialize sequences
        self.Filtered_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Filtered_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
        self.Filtered_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.Filtered_Residuals = torch.empty((self.BatchSize, T, self.n, 1))

        self.Kalman_Gains = torch.empty((self.BatchSize, T, self.m, self.n))

        self.Predicted_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Predicted_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.Predicted_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
        self.Predicted_Observation_Covariances = torch.empty((self.BatchSize, T, self.n, self.n))
        self.Predicted_Residuals = torch.zeros((self.BatchSize, T, self.n, 1))

        self.F_arr = torch.empty((self.BatchSize, T, self.m, self.m))
        self.H_arr = torch.empty((self.BatchSize, T, self.n, self.m))

        # Initialize time
        self.t = 0

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
        self.Filtered_Observation_Means[:, self.t] = self.h_batch(self.Filtered_State_Mean, self.t)
        self.Filtered_Residuals[:, self.t] = self.Filtered_Residual

        self.Kalman_Gains[:, self.t] = self.KG

        self.Predicted_State_Means[:, self.t] = self.Predicted_State_Mean
        self.Predicted_State_Covariances[:, self.t] = self.Predicted_State_Covariance
        self.Predicted_Observation_Means[:, self.t] = self.Predicted_Observation_Mean
        self.Predicted_Observation_Covariances[:, self.t] = self.Predicted_Observation_Covariance
        self.Predicted_Residuals[:, self.t] = self.Predicted_Residual

        self.F_arr[:, self.t] = self.F
        self.H_arr[:, self.t] = self.H

        self.t += 1

        return self.Filtered_State_Mean, self.Filtered_State_Covariance

    def ml_update_q(self, observation: torch.Tensor) -> None:
        """
        Online upate the estimate for the process noise, using ML estimation
        :param observation: Tensor of current observations
        :return: None
        """

        # Get the predicted residuals of the last nResidual time steps
        rho = self.Predicted_Residuals[:, max(self.t - self.nResiduals, 0): self.t + 1]

        rho_latest = (observation - self.Filtered_State_Mean).unsqueeze(0)

        rho_mean = torch.cat((rho, rho_latest), dim=1).mean(1)

        R_diag = torch.diagonal(self.get_R(self.t), dim1=-1, dim2=-2)

        P_diag = torch.diagonal(self.Filtered_State_Covariance, dim1=-1, dim2=-2)

        Q = torch.bmm(rho_mean.mT, rho_mean) / self.m - R_diag.mean() - P_diag.mean()
        # Q = torch.diagonal(torch.bmm(rho_mean, rho_mean.mT), dim1=-1, dim2=-2) / self.m - R_diag - P_diag

        self.update_Q(torch.clip(Q, 0) * torch.eye(self.m))


class KalmanSmoother(KalmanFilter):

    def __init__(self, sys_model: BaseSystemModel, em_vars: tuple = ('R', 'Q')):

        super(KalmanSmoother, self).__init__(sys_model, em_vars)

    def sgain(self, t: int) -> None:
        """
        Calculate Kalman smoother gain
        :param t: Time index
        :return: None
        """

        self.update_jacobians(t)

        self.SG = torch.bmm(self.F.mT, torch.linalg.pinv(self.Predicted_State_Covariance))
        self.SG = torch.bmm(self.Filtered_State_Covariance, self.SG)

    def scorrect(self):

        self.Smoothed_State_Mean = torch.bmm(self.SG, (self.Smoothed_State_Mean - self.Predicted_State_Mean))
        self.Smoothed_State_Mean = self.Filtered_State_Mean + self.Smoothed_State_Mean

        covariance_residual = self.Smoothed_State_Covariance - self.Predicted_State_Covariance
        self.Smoothed_State_Covariance = torch.bmm(covariance_residual, self.SG.mT)
        self.Smoothed_State_Covariance = torch.bmm(self.SG, self.Smoothed_State_Covariance)
        self.Smoothed_State_Covariance = self.Filtered_State_Covariance + self.Smoothed_State_Covariance

    @torch.no_grad()
    def smooth(self, observations: torch.Tensor, T: int) -> (torch.Tensor, torch.Tensor):
        """
        Perform kalman smoothing on the given observations
        :param observations: Tensor of observations dimensions (batch_size, T , channels, 1)
        :param T: Time horizon
        :return: Smoothed state means
        """

        # Perform Kalman filtering
        self.filter(observations, T)

        # Allocate data buffers
        self.Smoothed_State_Means = torch.empty((self.BatchSize, T, self.m, 1))
        self.Smoothed_Observation_Means = torch.empty((self.BatchSize, T, self.n, 1))
        self.Smoothed_State_Covariances = torch.empty((self.BatchSize, T, self.m, self.m))
        self.SGains = torch.empty((self.BatchSize, T - 1, self.m, self.m))

        # Initialize current smoothed state and covariance
        self.Smoothed_State_Mean = self.Filtered_State_Means[:, -1]
        self.Smoothed_State_Covariance = self.Filtered_State_Covariances[:, -1]

        # Set first smoothed state and covariance to be last filtered state
        self.Smoothed_State_Means[:, -1] = self.Smoothed_State_Mean
        self.Smoothed_State_Covariances[:, -1] = self.Smoothed_State_Covariance

        # Loop backwards in time
        for t in reversed(range(T - 1)):
            # Initialize current estimates
            self.Filtered_State_Mean = self.Filtered_State_Means[:, t]
            self.Predicted_State_Mean = self.Predicted_State_Means[:, t + 1]
            self.Filtered_State_Covariance = self.Filtered_State_Covariances[:, t]
            self.Predicted_State_Covariance = self.Predicted_State_Covariances[:, t + 1]

            # Calculate backwards kalman gain
            self.sgain(t)
            # Perform the smoothing
            self.scorrect()

            # Fill up buffers
            self.Smoothed_State_Means[:, t] = self.Smoothed_State_Mean
            self.Smoothed_Observation_Means[:, t] = self.h_batch(self.Smoothed_State_Mean, t)
            self.Smoothed_State_Covariances[:, t] = self.Smoothed_State_Covariance
            self.SGains[:, t] = self.SG

        return self.Smoothed_State_Means, self.Smoothed_State_Covariances

    def smooth_pair(self, T: int) -> None:
        """
        Calculate pairwise covariance between time steps
        :param T: Time horizon
        :return: None
        """
        self.Pairwise_Covariances = torch.zeros((self.BatchSize, T, self.m, self.m))

        for t in range(1, T):
            self.Smoothed_State_Covariance = self.Smoothed_State_Covariances[:, t]
            self.SG = self.SGains[:, t - 1]

            self.Pairwise_Covariances[:, t] = torch.bmm(self.Smoothed_State_Covariance, self.SG.mT)

    @torch.no_grad()
    def em(self,
           observations: torch.Tensor,
           T: int,
           q_2_init: float or None,
           r_2_init: float or None,
           num_its: int = 20,
           states: torch.Tensor = None,
           convergence_threshold=1e-6,
           smoothing_window_Q=-1,
           smoothing_window_R=-1) -> None or torch.Tensor:
        """
        Perform Expectation-Maximization on the parameters specified in the declaration
        :param observations: Tensor of observations
        :param T: Time horizon
        :param q_2_init: Initial estimate for the state covariance gain
        :param r_2_init: Initial estimate for the observation covariance gain
        :param num_its:  Maximum number of iterations
        :param states: (optional) Ground truth labels
        :param convergence_threshold: Threshold for the absolute change between consecutive estimates
        :param smoothing_window_Q: Window size for smoothing Q estimates
        :param smoothing_window_R: Window size for smoothing R estimates
        :return: None
        """

        # Set initial covariance estimates
        if q_2_init is not None:
            self.update_Q(q_2_init * torch.eye(self.m))
        if r_2_init is not None:
            self.update_R(r_2_init * torch.eye(self.n))

        # Set up iteration counter
        # iteration_counter = trange(num_its, desc='EM optimization steps')

        # If labels are available, calculate the loss
        losses = []
        loss_fn = torch.nn.MSELoss(reduction='mean')
        if states != None:
            states = states.squeeze()

        # Start iteration
        for n in range(num_its):

            # E-Step
            self.smooth(observations, T)

            # Calculate consecutive error covariances
            self.smooth_pair(T)

            ########################
            # Calculate expectations
            ########################

            # Expected covariance between hidden states
            self.U_xx = torch.einsum('BTmp,BTpn->BTmn', (self.Smoothed_State_Means, self.Smoothed_State_Means.mT))
            self.U_xx += self.Smoothed_State_Covariances

            # Expected covariance between hidden states and observations
            self.U_yx = torch.einsum('BTnp,BTpm->BTnm', (self.Observations, self.Smoothed_State_Means.mT))

            # Expected covariance between observations
            self.U_yy = torch.einsum('BTmp,BTpn->BTmn', (self.Observations, self.Observations.mT))

            # Expected covariance between hidden states except the last state
            self.V_xx = self.U_xx[:, :-1]

            # Expected covariance between consecutive hidden states
            self.V_x1x = torch.einsum('BTmp,BTpn->BTmn',
                                      (self.Smoothed_State_Means[:, 1:], self.Smoothed_State_Means[:, :-1].mT))
            self.V_x1x += self.Pairwise_Covariances[:, 1:]

            # Expected covariance between hidden states except the first state
            self.V_x1x1 = self.U_xx[:, 1:]

            # Update all variables specified
            for EmVar in self.em_vars:

                if EmVar == 'Q':
                    self.em_update_Q(smoothing_window_Q)

                elif EmVar == 'R':
                    self.em_update_R(smoothing_window_R)

                else:
                    self.__getattribute__(f'em_update_{EmVar}')()

            # Update iteration counter
            if states != None:
                loss = loss_fn(self.h_batch(self.Smoothed_State_Means.squeeze(), 0).squeeze(), states.squeeze())
                losses.append(10 * torch.log10(loss))
                # iteration_counter.set_description('EM Iteration loss: {} [dB]'.format(10 * torch.log10(loss).item()))

            # Check for convergence
            if all([self.__getattribute__(f'{i}_diff') < convergence_threshold for i in self.em_vars]):
                # print('Converged')
                break

        if states != None:
            return torch.tensor(losses)

    def em_update_H(self) -> None:
        """
        Update observation function
        :return: None
        """

        H_arr = torch.einsum('BTmp,BTpn->BTmn', (self.U_yx, torch.linalg.pinv(self.U_xx))).squeeze()

        # Try to calculate the difference between consecutive updates
        try:
            self.H_diff = torch.abs(torch.mean(H_arr - self.H_arr))
        except:
            self.H_diff = torch.inf

        self.H_em = True
        self.H_arr = H_arr

    def em_update_Mu(self) -> None:
        """
        Update the initial distribution mean
        :return: None
        """

        self.Mu_diff = torch.abs(torch.mean(self.Initial_State_Mean - self.Smoothed_State_Means[:, 0]))

        self.init_mean(self.Smoothed_State_Means[:, 0])

    def em_update_Sigma(self) -> None:
        """
       Update the initial distribution covariance
       :return: None
       """
        self.Sigma_diff = torch.abs(torch.mean(self.Initial_State_Covariance - self.Smoothed_State_Covariances[:, 0]))
        self.init_covariance(self.Smoothed_State_Covariances[:, 0])

    def em_update_R(self, smoothing_window: int = -1) -> None:
        """
        Update observation covariance estimate
        :param smoothing_window: Size of the smoothing window (-1: full average, 0: no average)
        :return: None
        """

        # Don't average the estimate
        if smoothing_window == 0:
            HU_xy = torch.einsum('BTmp,BTpn->BTmn', (self.H_arr, self.U_yx.mT))

            HUH = torch.einsum('BTmp,BTpk,BTkn->BTmn', (self.H_arr, self.U_xx, self.H_arr.mT))
            R_arr = self.U_yy - HU_xy - HU_xy.mT + HUH

        # Average over entire time horizon
        elif smoothing_window == -1:

            U_yx = self.U_yx.mean(1)

            U_xx = self.U_xx.mean(1)

            U_yy = self.U_yy.mean(1)

            H = self.H_arr.mean(1)

            HU_xy = torch.einsum('Bmp,Bpn->Bmn', (H, U_yx.mT))

            HUH = torch.einsum('Bmp,Bpk,Bkn->Bmn', (H, U_xx, H.mT))
            R_arr = U_yy - HU_xy - HU_xy.mT + HUH

            R_arr = R_arr.repeat(1, self.ssModel.T, 1, 1)

        # Average over given window size
        else:

            U_yx = moving_average(self.U_yx, window_size=smoothing_window)

            U_xx = moving_average(self.U_xx, window_size=smoothing_window)

            U_yy = moving_average(self.U_yy, window_size=smoothing_window)

            HU_xy = torch.einsum('BTmp,BTpn->BTmn', (self.H_arr, U_yx.mT))

            HUH = torch.einsum('BTmp,BTpk,BTkn->BTmn', (self.H_arr, U_xx, self.H_arr.mT))
            R_arr = U_yy - HU_xy - HU_xy.mT + HUH

        try:
            self.R_diff = torch.abs(torch.mean(R_arr.mean(1) - self.R_arr))
        except:
            self.R_diff = torch.inf

        # To numerically guarantee symetry
        self.R_arr = 0.5 * R_arr + 0.5 * R_arr.mT

    def em_update_F(self) -> None:
        """
        Update process jacobian
        :return: None
        """

        F_arr = torch.einsum('Bmp,Bpn->Bmn', (self.V_x1x.mean(1), torch.linalg.pinv(self.V_xx.mean(1))))

        try:
            self.F_diff = torch.abs(torch.mean(F_arr - self.F_arr))
        except:
            self.F_diff = torch.inf

        self.F_em = True
        self.F_arr = F_arr

    def em_update_Q(self, smoothing_window: int = -1) -> None:
        """
        Update process noise covariance
        :param smoothing_window: Size of the smoothing window (-1: full average, 0: no average)
        :return: None
        """
        # Don't average the estimate
        if smoothing_window == 0:

            FV_xx1 = torch.einsum('Bmp,BTpn->BTmn', (self.F_arr, self.V_x1x.mT))

            FVF = torch.einsum('Bmp,BTpk,Bkn->BTmn', (self.F_arr, self.V_xx, self.F_arr.mT))
            Q_arr = self.V_x1x1 - FV_xx1 - FV_xx1.mT + FVF
            Q_arr = torch.cat((Q_arr, Q_arr[:, 0].unsqueeze(1)), dim=1)

        # Average over entire time horizon
        elif smoothing_window == -1:

            V_x1x = self.V_x1x.mean(1)
            V_xx = self.V_xx.mean(1)

            V_x1x1 = self.V_x1x1.mean(1)

            F = self.F_arr.mean(1)

            FV_xx1 = torch.einsum('Bmp,Bpn->Bmn', (F, V_x1x.mT))

            FVF = torch.einsum('Bmp,Bpk,Bkn->Bmn', (F, V_xx, F.mT))
            Q_arr = V_x1x1 - FV_xx1 - FV_xx1.mT + FVF
            Q_arr = torch.clip(Q_arr, 0)
            Q_arr = Q_arr.repeat(1, self.ssModel.T, 1, 1)

        # Average over given window size
        else:

            V_x1x = moving_average(self.V_x1x, window_size=smoothing_window)

            V_xx = moving_average(self.V_xx, window_size=smoothing_window)

            V_x1x1 = moving_average(self.V_x1x1, window_size=smoothing_window)

            FV_xx1 = torch.einsum('BTmp,BTpn->BTmn', (self.F_arr[:, 1:], V_x1x.mT))

            FVF = torch.einsum('BTmp,BTpk,BTkn->BTmn', (self.F_arr[:, 1:], V_xx, self.F_arr[:, 1:].mT))
            Q_arr = V_x1x1 - FV_xx1 - FV_xx1.mT + FVF

            Q_arr = torch.cat((Q_arr[:, 0].unsqueeze(1), Q_arr), dim=1)

        try:
            self.Q_diff = torch.abs(torch.mean(Q_arr - self.Q_arr))
        except:
            self.Q_diff = torch.inf

        # To numerically guarantee symetry
        self.Q_arr = 0.5 * Q_arr + 0.5 * Q_arr.mT
