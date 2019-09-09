from particles import qmc

from numbers import Number
import torch
from torch import FloatTensor
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits
from torch.distributions.distribution import Distribution
#import numpy as np


class Uniform_RQMC(Distribution):
    r"""
    RQMC sampled uniform random variable.
    """
    arg_constraints = {'low': constraints.dependent, 'high': constraints.dependent}
    has_rsample = True

    def __init__(self, low, high, dim=1, scrambled=1, validate_args=None):
        self.low, self.high = broadcast_all(low, high)
        self.dim = dim
        self.scrambled = scrambled

        if isinstance(low, Number) and isinstance(high, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super(Uniform_RQMC, self).__init__(batch_shape, validate_args=validate_args)

        if self._validate_args and not torch.lt(self.low, self.high).all():
            raise ValueError("Uniform_RQMC is not defined when low>= high")

    def set_parameters(self, low, high):
        self.low, self.high = broadcast_all(low, high)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        n_samples = int(torch.prod(torch.tensor(shape)))
        if n_samples == 1:
            print("Warning: RQMC sample size should be greater than 1.")
        rand = qmc.sobol(N=n_samples, dim=self.dim, scrambled=self.scrambled) # scrambled=0
        #rand = rqmc_py.random_sequence_rqmc(size_mv=self.dim, i=0, n=n_samples)
        # rand = qmc_py.sobol_sequence(N=n_samples, DIMEN=self.dim, IFLAG=1,
        #                              iSEED=np.random.randint(10**5))  # .transpose()
        if self.dim == 1:
            rand = FloatTensor(rand).reshape(shape)
        else:
            rand = FloatTensor(rand).reshape(shape + torch.Size([self.dim]))

        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = value.ge(self.low).type_as(self.low)
        ub = value.lt(self.high).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    # TODO: add functions from binomial (e.g. new_)
    # TODO: How to deal with relative imports...
    # TODO: Deal with multi-dim sample shape / event_shape
    # TODO: What is the right way to deal if sample_shape is multi-dim

if __name__ == "__main__":
    dist = Uniform_RQMC(0, 1, dim=5)
    print(dist.sample(torch.Size([10])))
