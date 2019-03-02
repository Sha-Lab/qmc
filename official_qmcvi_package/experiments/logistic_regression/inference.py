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


class Inference_Reparam_Gradient():

    def __init__(self, model, learning_rate, use_RQMC):
        self.model = model
        self.learning_rate = learning_rate
        self.use_RQMC = use_RQMC

    def compute_loss(self, num_particles):
        '''
        Negative ELBO.
        '''
        theta = self.model.sample_guide_reparam(num_particles, self.use_RQMC)

        elbo = self.model.logprob_model(theta, self.model.data) - self.model.logprob_guide(theta)
        loss = -torch.mean(elbo, dim=0)
        loss_variance = torch.var(elbo, dim=0)

        return loss, loss_variance

    def do_inference(self, n_steps, num_particles, verbose=100):
        loss_values = []
        loss_variance_values = []
        loc_values = []
        scale_values = []

        optimizer = torch.optim.Adam(
            [self.model.loc_guide, self.model.scale_guide], lr=self.learning_rate)

        # optimize ELBO
        for step in range(n_steps):
            # inference step
            loss, loss_variance = self.compute_loss(num_particles)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store values
            loss_values.append(loss.item())
            loss_variance_values.append(loss_variance.item())
            loc_values.append(deepcopy(self.model.loc_guide))
            scale_values.append(deepcopy(self.model.scale_guide))

            # print elbo
            if verbose > 0 and step % verbose == 0:
                print('step: %d | loss: %.3f' % (step, loss.item()))

        return loss_values, loss_variance_values, torch.stack(loc_values), torch.stack(scale_values)


class Inference_Score_Gradient():

    def __init__(self, model, learning_rate, use_RQMC):
        self.model = model
        self.learning_rate = learning_rate
        self.use_RQMC = use_RQMC

    def compute_loss_and_set_gradients(self, num_particles):
        '''
        Negative ELBO.
        '''
        with torch.no_grad():
            theta = self.model.sample_guide_score(num_particles, self.use_RQMC)
            elbo_no_grad = self.model.logprob_model(
                theta, self.model.data) - self.model.logprob_guide(theta)

        # compute elbo
        elbo_with_grad = self.model.logprob_model(
            theta, self.model.data) - self.model.logprob_guide(theta)
        loss = -torch.mean(elbo_with_grad, dim=0)
        loss_variance = torch.var(elbo_with_grad, dim=0)

        # compute gradients with respect to following variables
        variables = [self.model.loc_guide, self.model.scale_guide]

        # manually compute the score function gradient
        score_function_part = -(self.model.logprob_guide(theta) * elbo_no_grad).mean(dim=0)

        with torch.no_grad():
            # score_gradient = torch.stack(grad(loss, variables))
            score_gradient = torch.stack(grad(score_function_part, variables)) * \
                loss.item() + torch.stack(grad(loss, variables))

        # manually set gradient values of variabels
        variables[0].grad = score_gradient[0].data
        variables[1].grad = score_gradient[1].data

        return loss, loss_variance

    def do_inference(self, n_steps, num_particles, verbose=100):
        loss_values = []
        loss_variance_values = []
        loc_values = []
        scale_values = []

        optimizer = torch.optim.Adam(
            [self.model.loc_guide, self.model.scale_guide], lr=self.learning_rate)

        # optimize ELBO
        for step in range(n_steps):
            # inference step
            optimizer.zero_grad()
            loss, loss_variance = self.compute_loss_and_set_gradients(num_particles)
            optimizer.step()

            # store values
            loss_values.append(loss.item())
            loss_variance_values.append(loss_variance.item())
            loc_values.append(deepcopy(self.model.loc_guide))
            scale_values.append(deepcopy(self.model.scale_guide))

            # print elbo
            if verbose > 0 and step % verbose == 0:
                print('step: %d | loss: %.3f' % (step, loss.item()))

        return loss_values, loss_variance_values, torch.stack(loc_values), torch.stack(scale_values)
