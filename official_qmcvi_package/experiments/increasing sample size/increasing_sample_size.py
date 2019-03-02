import sys
sys.path.append('../../')  # append rqmc distributions module

import torch
import torch.distributions as dist
from torch.distributions import Normal
from rqmc_distributions import Normal_RQMC
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy


class Inference():

    def __init__(self, loc_true, scale_true, learning_rate, use_RQMC):
        # true parameters
        self.loc_true = torch.tensor(loc_true, requires_grad=False)
        self.scale_true = torch.tensor(scale_true, requires_grad=False)

        # variational parameters (which are optimized)
        self.loc_guide = torch.zeros_like(self.loc_true, requires_grad=True)
        self.scale_guide = torch.ones_like(self.scale_true, requires_grad=True)

        # MC vs RQMC
        self.use_RQMC = use_RQMC

        # optimization
        self.learning_rate = learning_rate

    def logprop_model(self, z):
        logprob = torch.sum(dist.normal.Normal(self.loc_true, self.scale_true).log_prob(z), dim=1)
        return logprob

    def logprob_guide(self, z):
        logprob = torch.sum(dist.normal.Normal(self.loc_guide, self.scale_guide).log_prob(z), dim=1)
        return logprob

    def sample_guide_MC(self, num_particles):
        eps = Normal(torch.zeros_like(self.loc_guide), torch.ones_like(
            self.scale_guide)).sample(torch.Size([num_particles]))
        z = torch.mul(eps, self.scale_guide) + self.loc_guide
        return z

    def sample_guide_RQMC(self, num_particles):
        eps = Normal_RQMC(torch.zeros_like(self.loc_guide), torch.ones_like(
            self.scale_guide)).sample(torch.Size([num_particles]))
        z = torch.mul(eps, self.scale_guide) + self.loc_guide
        return z

    def compute_loss(self, num_particles):
        '''
        Negative ELBO.
        '''
        if self.use_RQMC:
            z = self.sample_guide_RQMC(num_particles)
        else:
            z = self.sample_guide_MC(num_particles)

        elbo = self.logprop_model(z) - self.logprob_guide(z)
        loss = -torch.mean(elbo, dim=0)
        loss_variance = torch.var(elbo, dim=0)

        return loss, loss_variance

    def do_inference(self, n_steps, verbose=100):
        loss_values = []
        loss_variance_values = []
        loc_values = []
        scale_values = []
        num_particles_values = []

        optimizer = torch.optim.SGD([self.loc_guide, self.scale_guide], lr=self.learning_rate)

        # optimize ELBO
        for step in range(n_steps):
            # compute number of particles
            num_particles = self.sample_size_schedule(step, n_steps, 10, n_steps)

            # inference step
            loss, loss_variance = self.compute_loss(num_particles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store values
            loss_values.append(loss.item())
            loss_variance_values.append(loss_variance.item())
            loc_values.append(deepcopy(self.loc_guide))
            scale_values.append(deepcopy(self.scale_guide))
            num_particles_values.append(num_particles)

            # print elbo
            if verbose > 0 and step % verbose == 0:
                print('step: %d | loss: %.3f' % (step, loss.item()))

        return loss_values, loss_variance_values, torch.stack(loc_values), torch.stack(scale_values), num_particles_values

    def sample_size_schedule(self, t, num_iters_saturation, start_sample_size, final_sample_size):
        """
        Increasing Sample Size Schedule
        t = index iteration
        num_iters_saturation = number of iterations when final sample size is reached
        start_sample_size = start sample size (at t = 0)
        final_sample_size = sample size at t = num_iters_saturation
        """
        if t >= num_iters_saturation:
            return final_sample_size

        # exponential growth
        a = start_sample_size / np.e
        b = (np.log(float(final_sample_size)/start_sample_size) + 1) / (num_iters_saturation - 1)

        return max(start_sample_size, int(a * np.exp(b * t)))


if __name__ == '__main__':
    torch.manual_seed(99)

    n_steps = 1000
    learning_rate = 0.01
    dim = 2
    loc_true = 1.2*torch.ones(dim)  # torch.tensor([0.5, -0.8, 0.1])
    scale_true = 0.6*torch.ones(dim)  # torch.tensor([1.5, 0.7, 0.9])

    # do inference MC
    print(" START INFERENCE MC")
    inference_mc = Inference(loc_true, scale_true, learning_rate, use_RQMC=False)
    loss_values_mc, loss_variance_values_mc, loc_values_mc, scale_values_mc, num_particles_values = inference_mc.do_inference(
        n_steps=n_steps, verbose=100)

    # do inference RQMC
    print(" START INFERENCE RQMC")
    inference_rqmc = Inference(loc_true, scale_true, learning_rate, use_RQMC=True)
    loss_values_rqmc, loss_variance_values_rqmc, loc_values_rqmc, scale_values_rqmc, num_particles_values = inference_rqmc.do_inference(
        n_steps=n_steps, verbose=100)

    # compute errors of param estimates
    error_loc_mc = torch.norm(loc_values_mc - loc_true, dim=1)**2
    error_scale_mc = torch.norm(scale_values_mc - scale_true, dim=1)**2
    error_loc_rqmc = torch.norm(loc_values_rqmc - loc_true, dim=1)**2
    error_scale_rqmc = torch.norm(scale_values_rqmc - scale_true, dim=1)**2

    # plot
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.semilogy(loss_values_mc, 'b-', alpha=0.7, label='MC')
    plt.semilogy(loss_values_rqmc, 'r-', alpha=0.7, label='RQMC')
    plt.legend()
    plt.title("Noisy neg. ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.subplot(2, 2, 2)
    plt.semilogy(error_loc_mc.data.numpy(), 'b-', alpha=0.8, label='MC: loc')
    plt.semilogy(error_scale_mc.data.numpy(), 'b--', alpha=0.8, label='MC: scale')
    plt.semilogy(error_loc_rqmc.data.numpy(), 'r-', alpha=0.8, label='RQMC: loc')
    plt.semilogy(error_scale_rqmc.data.numpy(), 'r--', alpha=0.8, label='RQMC: scale')
    plt.legend()
    plt.title("Error paramaters")
    plt.xlabel("step")
    plt.ylabel("MSE")

    plt.subplot(2, 2, 3)
    plt.semilogy(loss_variance_values_mc, 'b-', alpha=0.7, label='MC')
    plt.semilogy(loss_variance_values_rqmc, 'r-', alpha=0.7, label='RQMC')
    plt.legend()
    plt.title("Variance ELBO")
    plt.xlabel("step")
    plt.ylabel("variance")

    plt.subplot(2, 2, 4)
    # plt.plot(loc_grads.data.numpy(), 'b--', label='MC: scale')
    plt.semilogy(num_particles_values, 'k-')
    plt.title("Number of Particles")
    plt.xlabel("step")
    plt.ylabel("num particles")

    plt.savefig('plot_increasing.png')
    plt.close()
