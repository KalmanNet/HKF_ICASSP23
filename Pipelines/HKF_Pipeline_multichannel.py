import torch
from Filters.IntraHKF import IntraHKF
from Filters.InterHKF_multi_channel import InterHKF
from PriorModels.BasePrior import BasePrior
from SystemModels.BaseSysmodel import BaseSystemModel
from Filters.KalmanSmoother import KalmanFilter, KalmanSmoother

from Pipelines.Base_HKF_Pipeline import HKF_Pipeline


class Multi_channel_HKF_Pipeline(HKF_Pipeline):

    def __init__(self, prior_model: BasePrior, pat: int):
        super(Multi_channel_HKF_Pipeline, self).__init__(prior_model, pat)

    def init_filters(self, sys_model: BaseSystemModel, em_vars: tuple, test_set_length: int) \
            -> (KalmanSmoother, KalmanFilter or list):
        # Initialize the internal smoother and the external filters
        intra_HKF = IntraHKF(sys_model, self.em_vars)
        inter_HKFs = [InterHKF(self.num_channels, self.em_vars) for _ in range(self.T)]

        for inter_HKF in inter_HKFs:
            inter_HKF.init_online(test_set_length)

        return intra_HKF, inter_HKFs

    def inter_operation(self, inter_hkf: KalmanFilter or list, observation: torch.Tensor,
                        covariances: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        inter_filter_means = []
        inter_filter_covariances = []
        for timestep, inter_HKF in enumerate(inter_hkf):
            # Get internal smoother output as new input for the outer filter
            channel_smoother_mean = observation[timestep].reshape(self.num_channels, 1)
            # Get internal smoother covariance as estimate for the observation noise
            channel_smoother_covariance = covariances[timestep]

            # Update \mathcal{R}_\tau using smoother error covariance
            inter_HKF.update_R(channel_smoother_covariance)

            # ML update \mathcal{Q}_\tau
            inter_HKF.ml_update_q(channel_smoother_mean)

            # Get the output of the external KF
            inter_filter_mean, inter_filter_covariance = inter_HKF.update_online(channel_smoother_mean)

            inter_filter_means.append(inter_filter_mean.reshape(-1, self.num_channels))
            inter_filter_covariances.append(inter_filter_covariance)

        return torch.stack(inter_filter_means).squeeze(), torch.stack(inter_filter_covariances).squeeze()

    def intra_operation(self, intra_hkf: KalmanSmoother, observation: torch.Tensor, T: int, m: int) -> (torch.Tensor, torch.Tensor):

        # Smooth internally with learned parameters
        smoother_means, smoother_covariances = intra_hkf.smooth(observation, T)

        # Set up means and covariances
        smoother_means = smoother_means.reshape(T, -1)
        smoother_covariances = smoother_covariances.reshape(T, m, m)

        return smoother_means, smoother_covariances