from Filters.KalmanSmoother import KalmanFilter
import torch


class InterHKF(KalmanFilter):

    def __init__(self, T: int, em_vars=('R', 'Q'), n_residuals=5):
        self.m = T
        self.n = T
        self.Q = torch.eye(T)
        super(InterHKF, self).__init__(sys_model=None, em_vars=em_vars, n_residuals=n_residuals)

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
        self.KG = torch.linalg.pinv(self.Predicted_Observation_Covariance) # Assumes Q,R,P to be diagonal
        # self.KG = torch.diag_embed(self.KG)
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
        self.R = R

    def update_Q(self, Q: torch.Tensor) -> None:
        self.Q = Q


    def update_online(self, observations: torch.Tensor) -> torch.Tensor:
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

        return self.Filtered_State_Mean

