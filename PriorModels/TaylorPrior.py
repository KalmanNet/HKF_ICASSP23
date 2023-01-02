# _____________________________________________________________
# author: T. Locher, tilocher@ethz.ch
# _____________________________________________________________

from scipy.special import factorial
import torch
import numpy as np
from SystemModels.ExtendedSysmodel import ExtendedSystemModel
from PriorModels.BasePrior import BasePrior
from torch.nn.functional import pad

class TaylorPrior(BasePrior):

    def __init__(self, **kwargs):

        super(TaylorPrior, self).__init__()

        # Get Taylor model parameters
        self.taylor_order = kwargs.get('taylor_order', 5)
        self.delta_t = kwargs.get('delta_t', 1)
        self.channels = kwargs['channels']
        self.window_type = kwargs.get('window_type', 'rectangular')
        self.window_size = kwargs.get('window_size', 5)
        self.window_parameter = kwargs.get('window_parameter', 1)

        assert self.taylor_order >= 1, 'Taylor order must be at least 1'
        assert self.delta_t > 0, 'Time delta needs to be positive'

        self.basis_functions = np.array([[self.delta_t ** k / factorial(k)] for k in range(1, self.taylor_order + 1)])
        self.basis_functions = torch.from_numpy(self.basis_functions).float()

        self.derivative_coefficients = torch.ones(self.taylor_order,1)

        self.window_weights = self.create_window(self.window_type, self.window_size, self.window_parameter)

    def create_window(self, window_type: str, window_size: int, window_parameter: float) -> torch.Tensor:
        """
        Create the weights given by the window parameters
        :param window_type: The type of window used, can be 'rectangular',  'exponential', 'gaussian', 'linear'
        :param window_size: Size of the window
        :param window_parameter: Parameter corresponding to the window type, e.g. standard deviation for 'gaussian'
        :return: Weights of the window
        """

        half_window = int(window_size / 2)
        if window_type == 'rectangular':
            weights = torch.from_numpy(np.array([window_parameter for _ in range(window_size)]))

        elif window_type == 'exponential':
            weights = np.array([window_parameter ** (np.abs(w - half_window)) for w in range(window_size)])
            weights = torch.from_numpy(weights)

        elif window_type == 'gaussian':
            weights = [window_parameter * np.exp(-(w-half_window)**2/(2*window_parameter)) for w in range(window_size)]
            weights = torch.tensor(weights)

        elif window_type == 'linear':
            slope = 1 / window_size
            weights = np.array([-slope*np.abs(w - half_window) + window_parameter for w in range(window_size)])
            weights = torch.from_numpy(weights)

        else:
            raise ValueError('Window not supported')

        return weights

    @torch.no_grad()
    def fit(self, data: torch.Tensor):

        # Get data dimensions
        batch_size, self.time_steps, self.channels = data.shape

        # Reshape basis functions to match data
        basis_functions = self.basis_functions.reshape((1, 1, -1))
        basis_functions = basis_functions.repeat((self.window_size, batch_size, 1))

        # Initialize coefficient buffer
        derivative_coefficients = torch.zeros(self.time_steps, self.taylor_order, self.channels)

        # Calculate the size of half a window
        half_window_size = int(self.window_size / 2)

        # Calculate the amount of padding necessary
        lower_bound_index = - half_window_size
        upper_bound_index = half_window_size + 1

        # Pad data for the averaging
        padded_data = pad(data, (0, 0, -lower_bound_index, upper_bound_index), 'replicate')

        # Covariance matrix of the basis function
        covariance = torch.bmm(basis_functions.mT, basis_functions)

        # Solve T-LS problems
        for t in range(self.time_steps):

            # Get current observations and next observations
            current_state = padded_data[:, t-lower_bound_index-half_window_size:t+half_window_size-lower_bound_index+1]
            observations = padded_data[:, t-lower_bound_index-half_window_size+1:t+half_window_size-lower_bound_index+2]

            # Target for the LS-regression
            target_tensor = (observations - current_state).reshape(self.window_size, batch_size, -1)

            # Get the cross correlation for each time step
            cross_correlation = torch.bmm(basis_functions.mT, target_tensor)

            # Weight both the correlation and the covariance
            weights = self.window_weights.reshape(-1,1,1)
            weighted_covariance = (weights * covariance).sum(0)
            weighted_cross_correlation = (weights * cross_correlation).sum(0)

            # Perform regression
            derivatives_t = torch.mm(torch.linalg.pinv(weighted_covariance), weighted_cross_correlation)

            # Fill buffer
            derivative_coefficients[t] = derivatives_t

        self.derivative_coefficients = derivative_coefficients

        return derivative_coefficients

    def f(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        State evolution function of the taylor model
        :param x: Current state tensor
        :param t: Time step
        :return: Evolved state, based on the calculated coefficients
        """
        return x + torch.mm(self.derivative_coefficients[t].T, self.basis_functions)

    def f_jacobian(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        State Jacobian function of the system model
        :param x: Current state tensor
        :param t: Time step
        :return: State Jacobian
        """
        return torch.eye(self.channels)

    def h(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Observation function of the system model
        :param x: Current state tensor
        :param t: Time step
        :return: Observed state
        """
        return x

    def h_jacobian(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Observation Jacobian function of the system model
        :param x: Current state tensor
        :param t: Time step
        :return: Observation Jacobian
        """
        return torch.eye(self.channels)

    def get_sys_model(self) -> ExtendedSystemModel:
        """
        Create and return a ExtendedSystemModel instance
        :param T: Time horizon
        :return: System model based on the fitted data
        """
        sys_model = ExtendedSystemModel(self.f, 0, self.h, 0, self.time_steps, self.channels, self.channels)
        sys_model.set_f_jacobian(self.f_jacobian)
        sys_model.set_h_jacobian(self.h_jacobian)

        return sys_model
