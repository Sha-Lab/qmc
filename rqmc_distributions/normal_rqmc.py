from __future__ import absolute_import, division, print_function

from numbers import Number
import torch
import math

from rqmc_distributions.uniform_rqmc import Uniform_RQMC
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property, logits_to_probs, probs_to_logits
from torch.distributions.distribution import Distribution
#from pyro.distributions.torch_distribution import TorchDistribution


class Normal_RQMC(Distribution):
    r"""
    RQMC sampled normal random variable.
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        # print('normal_rqmc init')
        # TODO: dim should be one-dimensional array! How to deal with different shapes? (especially dim-less)
        dim = loc.shape[0]

        # init rqmc uniform random variable
        # print('dim ' + str(dim))
        self.u_rqmc = Uniform_RQMC(0, 1, dim)
        # distribution variables
        self.loc, self.scale = broadcast_all(loc, scale)

        # print(self.loc, self.scale)

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal_RQMC, self).__init__(batch_shape, validate_args=validate_args)

    def set_parameters(self, loc, scale):
        self.loc, self.scale = broadcast_all(loc, scale)

    def rsample(self, sample_shape=torch.Size()):
        # shape = self._extended_shape(sample_shape)
        shape = sample_shape  # TODO: what is extended shape doing?
        u = self.u_rqmc.sample(shape)
        # print(shape)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def icdf(self, value):
        '''
        TODO: Fixed inf bug by a dirty hack!
        We replace Inf of torch.erfinv by torch.erfinv(torch.tensor(0.99999997))

        We should implement our own icdf to fix the bug properly.
        '''
        if self._validate_args:
            self._validate_sample(value)
        res = self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)
        res = torch.clamp(res, -5.3, 5.3)
        #if torch.any(res == float('inf')) or torch.any(res == float('-inf')): import ipdb; ipdb.set_trace()
        return res

    # TODO: add functions from binomial (e.g. new_)
    # TODO: How to deal with relative imports...
    # TODO: Deal with multi-dim sample shape
    # TODO: What is the right way to deal if sample_shape is multi-dim
