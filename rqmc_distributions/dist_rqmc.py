# This is based on sobol_seq: https://github.com/naught101/sobol_seq
# Support generate rqmc sequence one-by-one
# The sequence is generated in numpy format
import sobol_seq
import numpy as np
from scipy.stats import norm


class Uniform_RQMC:
    def __init__(self, dim=1, scrambled=False):
        self.dim = dim
        self.scrambled = scrambled
        if scrambled:
            self.bias = np.random.rand(dim)
        self.seed = 1

    def sample(self, size):
        res = []
        for _ in range(size):
            vec, self.seed = sobol_seq.i4_sobol(self.dim, self.seed)
            res.append(vec)
        res = np.asarray(res)
        if self.scrambled: res = (res + self.bias) % 1.0
        return res

class Normal_RQMC:
    def __init__(self, dim=1, scrambled=False):
        self.sampler = Uniform_RQMC(dim, scrambled=scrambled)

    def sample(self, size):
        return norm.ppf(self.sampler.sample(size))
