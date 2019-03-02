import torch
from rqmc_distributions import Uniform_RQMC, Normal_RQMC

print("Test RQMC sampling.\n")

print("Sample uniform RQMC")
u = Uniform_RQMC(0, 1).sample(torch.Size([6]))
print(u)

print("\nSample normal RQMC")
x = Normal_RQMC(torch.tensor([0.]), torch.tensor([1.])).sample(torch.Size([6]))
print(x)

print("\nEverything seems to work fine!")
