from SystemModels.BaseSysmodel import BaseSystemModel


class ExtendedSystemModel(BaseSystemModel):

    def __init__(self, f: callable or str = 'Identity', q: float = 0, h: callable or str = 'Identity', r: float = 0,
                 T: int = 1, m: int = 1, n: int = 1):

        super(ExtendedSystemModel, self).__init__(q=q,
                                                  r=r,
                                                  T=T,
                                                  m=m,
                                                  n=n
                                                  )

        if not f == 'Identity':
            # Override state evolution function
            self.f = f
            # Set Jacobian to be computed numerically
            self.FJacSet = False

        if not h == 'Identity':
            # Override observation function
            self.h = h
            # Set Jacobian to be computed numerically
            self.HJacSet = False
