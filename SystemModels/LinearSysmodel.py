import torch
from SystemModels.BaseSysmodel import BaseSystemModel


class LinearSystemModel(BaseSystemModel):

    def __init__(self, F: torch.Tensor, q: int, H:torch.Tensor, r: int, T: int, m: int, n: int):

        super(LinearSystemModel, self).__init__(q=q,
                                                r=r,
                                                T=T,
                                                m=m,
                                                n=n
                                                )
        # Linear evolution and observation matrices
        self.F = F
        self.H = H

    def f(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Linear State evolution function
        :param x: State tensor x_{t}
        :param t: -
        :return: Linear evolution of state
        """
        return torch.mm(self.F, x)

    def h(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Linear observation function
        :param x: State tensor x_{t}
        :param t: -
        :return: Linear observation of state
        """
        return torch.mm(self.H, x)


    def f_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
         Analytical process evolution Jacobian function
         :param x: State tensor x_{t}
         :param t: Time index (if necessary)
         :return: State Jacobian
         """
        return self.F

    def h_jacobian(self, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Analytical observation Jacobian function
        :param x: State tensor x_{t}
        :param t: Time index (if necessary)
        :return: Observation Jacobian
        """
        return self.H
