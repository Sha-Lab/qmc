import sys
sys.path.append('../../')

from rqmc_distributions import Uniform_RQMC, Normal_RQMC
import torch
from torch.distributions import Uniform, Normal
import matplotlib.pyplot as plt
import numpy as np

n_samples = 200
shape = torch.Size([n_samples])
dim = 2
torch.manual_seed(999)

# params uniform distribution
u_min = -1
u_max = 1

# params normal distribution
loc = torch.tensor([0.5, 0.9])
scale = torch.tensor([0.7, 1.8])

# setup distributions
uniform = Uniform(u_min*torch.ones(dim), u_max*torch.ones(dim))
uniform_rqmc = Uniform_RQMC(u_min, u_max, dim)
normal = Normal(loc, scale)
normal_rqmc = Normal_RQMC(loc, scale)

# sample
u = uniform.sample(shape)
u_rqmc = uniform_rqmc.sample(shape)
x = normal.sample(shape)
x_rqmc = normal_rqmc.sample(shape)

# compute empirical mean and std
x_mean = torch.stack([torch.mean(x[:i, :], dim=0) for i in range(1, n_samples)])
x_std = torch.stack([torch.std(x[:i, :], dim=0) for i in range(1, n_samples)])
x_rqmc_mean = torch.stack([torch.mean(x_rqmc[:i, :], dim=0) for i in range(1, n_samples)])
x_rqmc_std = torch.stack([torch.std(x_rqmc[:i, :], dim=0) for i in range(1, n_samples)])


# plot
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.scatter(u[:, 0].data.numpy(), u[:, 1].data.numpy())
plt.title("MC Uniform")

plt.subplot(2, 3, 2)
plt.scatter(u_rqmc[:, 0].data.numpy(), u_rqmc[:, 1].data.numpy(), color='darkorange')
plt.title("RQMC Uniform")

plt.subplot(2, 3, 4)
plt.scatter(x[:, 0].data.numpy(), x[:, 1].data.numpy())
plt.title("MC Normal")

plt.subplot(2, 3, 5)
plt.scatter(x_rqmc[:, 0].data.numpy(), x_rqmc[:, 1].data.numpy(), color='darkorange')
plt.title("RQMC Normal")

plt.subplot(2, 3, 3)
plt.semilogy((torch.norm(x_mean - loc, dim=1)**2).data.numpy(), label='error mean MC')
plt.semilogy((torch.norm(x_rqmc_mean - loc, dim=1)**2).data.numpy(), label='error mean RQMC')
plt.legend()
plt.title("Error of empirical mean of normal distribution")

plt.subplot(2, 3, 6)
plt.semilogy((torch.norm(x_std - scale, dim=1)**2).data.numpy(), label='error std MC')
plt.semilogy((torch.norm(x_rqmc_std - scale, dim=1)**2).data.numpy(), label='error std RQMC')
plt.legend()
plt.title("Error of empirical std deviation of normal distribution")

# save fig
plt.savefig("./comparison_samplers.png")
