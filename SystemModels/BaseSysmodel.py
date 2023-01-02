import torch


class BaseSystemModel:

    def __init__(self, q: float = 0, r: float = 0, T: int = 1, m: int = 1, n: int = 1):
        """
        Base Class for System Models
        :param q: Process noise scalar
        :param r: Observation noise scalar
        :param T: Time horizon
        :param m: State-space dimension
        :param n: Observation-space dimension
        """

        ########################
        # Set Class attributes #
        ########################

        # Noise atts.
        self.q = q
        self.r = r

        # Time horizon
        self.T = T

        # System dimensions
        self.m = m  # Hidden state dimension
        self.n = n  # Observed state dimension

        ##################
        # Derived Values #
        ##################
        self.Q = q ** 2 * torch.eye(m)
        self.R = r ** 2 * torch.eye(n)

        #######################
        # Jacobian Parameters #
        #######################
        self.FJacSet = True
        self.HJacSet = True

        # Initialize the prior x_{0|0} ~ N(0,I)
        self.m1x_0 = torch.zeros((self.m, 1))
        self.m2x_0 = torch.eye(self.m)

    def f(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        State Evolution Function
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Evolved state tensor x_{t+1}
        """
        return x

    def f_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Analytical process evolution Jacobian function
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: State Jacobian
        """
        return torch.eye(self.m)

    def get_f_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Return the Jacobian for the current state and time index
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Jacobian of
        """
        # If analytical Jacobian is available
        if self.FJacSet:
            return self.f_jacobian(x, t)
        # Else compute numerically
        else:
            return torch.autograd.functional.jacobian(self.f, (x, torch.tensor(t, dtype=torch.float32)))[0]

    def set_f_jacobian(self, df: callable) -> None:
        """
        Function to set the Jacobian if there is an analytical solution
        :param df: Process Jacobian function
        :return: None
        """
        # Flag to not compute the Jacobian numerically
        self.FJacSet = True

        # Set the Jacobian function
        self.f_jacobian = df


    def h(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Observation Function
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Observation y_{t}
        """
        return x

    def h_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Analytical observation Jacobian function
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Observation Jacobian
        """
        return torch.eye(self.n)

    def get_h_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Return the observation Jacobian for the state and time index
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Jacobian
        """
        # If analytical Jacobian is available
        if self.HJacSet:
            return self.h_jacobian(x, t)
        # Else compute numerically
        else:
            return torch.autograd.functional.jacobian(self.h, (x, torch.tensor(t, dtype=torch.float32)))[0]


    def set_h_jacobian(self, dh: callable) -> None:
        """
        Function to set the Jacobian if there is an analytical solution
        :param dh: Observation Jacobian function
        :return: None
        """
        # Flag to not compute the Jacobian numerically
        self.HJacSet = True

        # Set the Jacobian function
        self.h_jacobian = dh

    def init_sequence(self, m1x_0: torch.Tensor, m2x_0: torch.Tensor) -> None:
        """
        Initialize Sequence prior
        :param m1x_0: 1st moment prior
        :param m2x_0: 2nd moment prior
        :return: None
        """

        # Set mean prior
        self.m1x_0 = m1x_0
        # Set covariance prior
        self.m2x_0 = m2x_0

    def update_covariance_gain(self, q: float, r: float) -> None:
        """ Update noise covariance gains
        update
        :param q: Process noise covariance gain
        :param r: Observation noise covariance gain
        :return: None
        """

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def update_R(self, R: torch.Tensor) -> None:
        """
        Update observation covariance
        :param R: Observation covariance tensor
        :return: None
        """
        self.R = R

    def update_Q(self, Q: torch.Tensor) -> None:
        """
        Update process covariance
        :param Q: Process covariance tensor
        :return: None
        """
        self.Q = Q

    def generate_sequence(self, T: int) -> tuple:
        """
        Markov Chain simulation of the state-space given current parameters
        :param T: Length of simulation
        :return: States and observations
        """
        # Pre allocate an array for current state
        self.x = torch.empty((T, self.m, 1))
        # Pre allocate an array for current observation
        self.y = torch.empty((T, self.n, 1))
        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        # Generate Sequence Iteratively
        for t in range(T):

            # State Evolution

            # Process Noise
            if self.q == 0:
                xt = self.f(self.x_prev, t)
            else:
                xt = self.f(self.x_prev, t)
                mean = torch.zeros([self.m])

                eq = torch.distributions.MultivariateNormal(loc=mean.reshape(1, -1),
                                                            covariance_matrix=self.Q).sample().reshape(self.m, 1)

                # Additive Process Noise
                xt = torch.add(xt, eq)

            # Observation function
            yt = self.h(xt, t)

            # Observation Noise
            mean = torch.zeros([self.n, 1])
            er = torch.normal(mean, self.r)

            # Additive Observation Noise
            yt = torch.add(yt, er)

            # Squeeze to Array

            # Save Current State to Trajectory Array
            self.x[t] = xt

            # Save Current Observation to Trajectory Array
            self.y[t] = yt

            # Save Current to Previous
            self.x_prev = xt

        return self.x, self.y

    def generate_batch(self, batchsize: int, T: int) -> tuple:
        """
        Function to generate a batch of simulations
        :param batchsize: Number of batches
        :param T: Length of the simulations
        :return: A tuple of hidden states and their observations
        """
        # Allocate Empty Array for Input
        self.Input = torch.empty(batchsize, T, self.n, 1)

        # Allocate Empty Array for Target
        self.Target = torch.empty(batchsize, T, self.m, 1)

        for i in range(batchsize):
            # Generate Sequence

            self.init_sequence(self.m1x_0, self.m2x_0)
            self.generate_sequence(T)

            # Training sequence input
            self.Input[i, :, :] = self.y

            # Training sequence output
            self.Target[i, :, :] = self.x

        return self.Input, self.Target
