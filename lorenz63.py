import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
import hydra_zen


def dyn_lorenz63(t, x, sigma=10., rho=28., beta=8./3):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(x[1]-x[0])
    x_2 = x[0]*(rho-x[2])-x[1]
    x_3 = x[0]*x[1] - beta*x[2]
    dx  = np.array([x_1, x_2, x_3])
    return dx


def trajectory_da(fn, y0, solver_kw, warmup_kw=None):
    if warmup_kw is not None:
        warmup = solve_ivp(fn, y0=y0, **{**solver_kw, **warmup_kw})
        y0 = warmup.y[:,-1]
    sol = solve_ivp(fn, y0=y0, **solver_kw)
    return xr.DataArray(sol.y, dims=('component', 'time'), coords={'component': ['x', 'y', 'z'], 'time': sol.t})


traj_cfg = hydra_zen.builds(
    trajectory_da,
    fn=hydra_zen.builds(dyn_lorenz63, zen_partial=True),
    y0=[8, 0, 30],
    solver_kw = dict(
        t_span=[0.01, 200 + 1e-6],
        t_eval=hydra_zen.builds(np.arange, start=0.01, stop= 200 + 1e-6, step=0.01),
        first_step=0.01,
        method='RK45',
    ),
    warmup_kw = dict(
        t_span=[0.01, 5 + 1e-6],
        t_eval=dict(_target_='numpy.arange', start=0.01, stop= 5 + 1e-6, step=0.01),
    )
)

def only_first_obs(da):
    new_da = xr.full_like(da, np.nan)
    new_da.loc['x']=da.loc['x']
    return new_da

def subsample(da, sample_step=20):
    new_da = xr.full_like(da, np.nan)
    new_da.loc[:, ::sample_step]=da.loc[:, ::sample_step]
    return new_da

def add_noise(da, sigma=2**.5):
    return da  + np.random.randn(*da.shape) * sigma


if __name__ == '__main__':
    import hydra
    l63_da = hydra.utils.call(traj_cfg)
    l63_da.sel(component='x').plot()
