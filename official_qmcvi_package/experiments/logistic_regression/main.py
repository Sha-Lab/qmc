import sys
sys.path.append('../../')  # append rqmc distributions module

import torch
import torch.distributions as dist
from torch.autograd import grad
from torch.distributions import Normal
from rqmc_distributions import Normal_RQMC
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from copy import deepcopy

from bayesian_logistic_regression import Bayesian_Logistic_Regression
from inference import Inference_Score_Gradient, Inference_Reparam_Gradient


if __name__ == '__main__':
    torch.manual_seed(400)

    # data
    N = 1000
    dim = 10
    theta_true = torch.randn(dim)

    # optimization
    n_steps = 1000
    num_particles_reparam = 10
    num_particles_score = 200
    learning_rate = 0.005

    # init inference methods with model
    inference_reparam_mc = Inference_Reparam_Gradient(
        Bayesian_Logistic_Regression(theta_true, N), learning_rate, use_RQMC=False)
    inference_reparam_rqmc = Inference_Reparam_Gradient(
        Bayesian_Logistic_Regression(theta_true, N), learning_rate, use_RQMC=True)
    inference_score_mc = Inference_Score_Gradient(
        Bayesian_Logistic_Regression(theta_true, N), learning_rate, use_RQMC=False)
    inference_score_rqmc = Inference_Score_Gradient(
        Bayesian_Logistic_Regression(theta_true, N), learning_rate, use_RQMC=True)

    # do inference MC
    print(" START INFERENCE WITH REPARAM GRAD USING MC")
    loss_values_reparam_mc, loss_variance_values_reparam_mc, loc_values_reparam_mc, scale_values_reparam_mc = inference_reparam_mc.do_inference(
        n_steps=n_steps, num_particles=num_particles_reparam, verbose=100)

    # do inference RQMC
    print(" START INFERENCE WITH REPARAM GRAD USING RQMC")
    loss_values_reparam_rqmc, loss_variance_values_reparam_rqmc, loc_values_reparam_rqmc, scale_values_reparam_rqmc = inference_reparam_rqmc.do_inference(
        n_steps=n_steps, num_particles=num_particles_reparam, verbose=100)

    # do inference MC
    print(" START INFERENCE WITH SCORE GRAD USING MC")
    loss_values_score_mc, loss_variance_values_score_mc, loc_values_score_mc, scale_values_score_mc = inference_score_mc.do_inference(
        n_steps=n_steps, num_particles=num_particles_reparam, verbose=100)

    # do inference RQMC
    print(" START INFERENCE WITH SCORE GRAD USING RQMC")
    loss_values_score_rqmc, loss_variance_values_score_rqmc, loc_values_score_rqmc, scale_values_score_rqmc = inference_score_rqmc.do_inference(
        n_steps=n_steps, num_particles=num_particles_reparam, verbose=100)

    # compute errors of theta estimates
    error_theta_reparam_mc = torch.norm(loc_values_reparam_mc - theta_true, dim=1)**2
    error_theta_reparam_rqmc = torch.norm(loc_values_reparam_rqmc - theta_true, dim=1)**2
    error_theta_score_mc = torch.norm(loc_values_score_mc - theta_true, dim=1)**2
    error_theta_score_rqmc = torch.norm(loc_values_score_rqmc - theta_true, dim=1)**2

    # plot
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.semilogy(loss_values_reparam_mc, 'b-', alpha=0.7, label='Reparam MC')
    plt.semilogy(loss_values_reparam_rqmc, 'r-', alpha=0.7, label='Reparam RQMC')
    plt.legend()
    plt.title("Reparam Gradient: Noisy neg. ELBO ({} particles)".format(num_particles_reparam))
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.subplot(2, 3, 2)
    plt.semilogy(loss_variance_values_reparam_mc, 'b-', alpha=0.7, label='Reparam MC')
    plt.semilogy(loss_variance_values_reparam_rqmc, 'r-', alpha=0.7, label='Reparam RQMC')
    plt.legend()
    plt.title("Reparam Gradient: Variance Gradient ({} particles)".format(num_particles_reparam))
    plt.xlabel("step")
    plt.ylabel("variance")

    plt.subplot(2, 3, 3)
    plt.semilogy(error_theta_reparam_mc.data.numpy(), 'b-', alpha=0.8, label='Reparam MC')
    plt.semilogy(error_theta_reparam_rqmc.data.numpy(), 'r-', alpha=0.8, label='Reparam RQMC')
    plt.legend()
    plt.title("Reparam Gradient: MAP estimate error ({} particles)".format(num_particles_reparam))
    plt.xlabel("step")
    plt.ylabel("MSE")

    plt.subplot(2, 3, 4)
    plt.semilogy(loss_values_score_mc, 'b-', alpha=0.7, label='Score MC')
    plt.semilogy(loss_values_score_rqmc, 'r-', alpha=0.7, label='Score RQMC')
    plt.legend()
    plt.title("Score Gradient: Noisy neg. ELBO ({} particles)".format(num_particles_score))
    plt.xlabel("step")
    plt.ylabel("loss")

    plt.subplot(2, 3, 5)
    plt.semilogy(loss_variance_values_score_mc, 'b-', alpha=0.7, label='Score MC')
    plt.semilogy(loss_variance_values_score_rqmc, 'r-', alpha=0.7, label='Score RQMC')
    plt.legend()
    plt.title("Score Gradient: Variance Gradient ({} particles)".format(num_particles_score))
    plt.xlabel("step")
    plt.ylabel("variance")

    plt.subplot(2, 3, 6)
    plt.semilogy(error_theta_score_mc.data.numpy(), 'b-', alpha=0.8, label='Score MC')
    plt.semilogy(error_theta_score_rqmc.data.numpy(), 'r-', alpha=0.8, label='Score RQMC')
    plt.legend()
    plt.title("Score Gradient: MAP estimate error ({} particles)".format(num_particles_score))
    plt.xlabel("step")
    plt.ylabel("MSE")

    plt.savefig('plot_logistic_regression.png')
    plt.close()
