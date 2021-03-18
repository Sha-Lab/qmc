
Part of our code base for combining a/rqmc with pg methods.

Files of interest:
- `main.py`: main entry point for learning (not only LQR, but for all environments and methods).
- `rqmc_distributions/normal_rqmc.py` and `rqmc_distributions/uniform_rqmc.py`: define PyTorch Distributions classes for RQMC sampling, built on top of SSJ (also see `rqmc_distributions/ssj_sobol.py` which defines the SSJ Java → Python wrapper).
- `envs/lqr.py`: implementation of the LQR environment, which is typically instantiated as follows:

```python
lqr = LQR(
    N=6,
        M=8,
        init_scale=3.0,
        max_steps=20,
        Sigma_s_kappa=1.0,
        Q_kappa=3.0,
        P_kappa=3.0,
        A_norm=1.0,
        B_norm=1.0,
        Sigma_s_scale=1.0,
        random_init=False,
        lims=100,
)
```
