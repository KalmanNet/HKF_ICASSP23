from Filters.KalmanSmoother import KalmanSmoother
from SystemModels.BaseSysmodel import BaseSystemModel
import torch


class IntraHKF(KalmanSmoother):
    """
    Intra heartbeat Kalman filter. Removes unnecessary computation with identity Jacobians
    """

    def __init__(self, ssModel: BaseSystemModel, em_vars: tuple = ('R', 'Q')):
        super(IntraHKF, self).__init__(ssModel, em_vars)

    def predict(self, t: int) -> None:
        """
        Prediction step
        :param t: Time index
        :return
        """
        # Predict the 1-st moment of x
        self.Predicted_State_Mean = self.f_batch(self.Filtered_State_Mean, t)

        # Compute Jacobians
        self.update_jacobians(t)

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
