import sys
sys.path.append('../../')  # append rqmc distributions module

import torch
import torch.distributions as dist
from torch.autograd import grad
from torch.distributions import Normal
from rqmc_distributions import Normal_RQMC
from pyro.optim import Adam, SGD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy


class Bayesian_Logistic_Regression():

    def __init__(self, theta_true, N):
        self.theta_true = theta_true

        # generate data
        self.data = self.generate_data(theta_true, N)

        # guide (variational distribution)
        self.loc_guide = torch.zeros_like(self.theta_true, requires_grad=True)
        self.scale_guide = torch.ones_like(self.theta_true, requires_grad=True)

    def logprob_model(self, theta, data):
        assert theta.dim() == 2  # theta.shape = (num_particles, dim)

        # unpack data
        X = data['X']
        y = data['y']
        N, dim = X.size()

        # prior
        log_prior = Normal(torch.zeros(dim), torch.ones(dim)).log_prob(theta).sum(dim=1)

        # likelihood
        log_likelihood = (torch.log(torch.sigmoid(
            y.view(-1, 1) * torch.matmul(X, theta.permute(1, 0))))).sum(dim=0)

        # joint
        return log_likelihood + log_prior

    def logprob_guide(self, theta):
        assert theta.dim() == 2  # theta.shape = (num_particles, dim)

        logprob = torch.sum(dist.normal.Normal(
            self.loc_guide, self.scale_guide).log_prob(theta), dim=1)
        return logprob

    def sample_guide_reparam(self, num_particles, use_RQMC):
        '''
        reparameterized sampling from the guide as used for the reparameterization gradient estimator
        '''
        if use_RQMC:
            eps = Normal_RQMC(torch.zeros_like(self.loc_guide), torch.ones_like(
                self.scale_guide)).sample(torch.Size([num_particles]))
        else:
            eps = Normal(torch.zeros_like(self.loc_guide), torch.ones_like(
                self.scale_guide)).sample(torch.Size([num_particles]))

        theta = torch.mul(eps, self.scale_guide) + self.loc_guide
        return theta

    def sample_guide_score(self, num_particles, use_RQMC):
        '''
        sampling from the guide directly as used for the score function gradient estimator
        '''
        if use_RQMC:
            theta = Normal_RQMC(self.loc_guide, self.scale_guide).sample(
                torch.Size([num_particles]))
        else:
            theta = Normal(self.loc_guide, self.scale_guide).sample(torch.Size([num_particles]))

        return theta

    def generate_data(self, theta_true, N):
        '''
        generate data from model
        '''
        dim = theta_true.size(0)
        X = torch.randn(N, dim)
        p = torch.sigmoid(torch.matmul(X, theta_true))
        y = dist.Binomial(1, p).sample()
        data = {'X': X, 'y': y}

        return data
