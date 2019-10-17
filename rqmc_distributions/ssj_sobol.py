import scipy
import numpy as np
from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
sobol_entry = gateway.entry_point

def uniform(power, dim):
    points = sobol_entry.sample(power, dim)
    return np.asarray([list(row) for row in list(points)])


if __name__ == "__main__":
    print(uniform(3, 5))
