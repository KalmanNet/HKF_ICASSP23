import torch


class BasePrior:

    def __init__(self, **kwargs):
        pass

    def fit(self, data: torch.Tensor):
        raise NotImplementedError

    def get_sys_model(self):
        raise NotImplementedError
