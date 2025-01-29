import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def vanderpol(x, y, mu):
    fx = (1 / 3 * x**3) - x
    dxdt = mu * (y - fx)
    dydt = (-1 / mu) * x
    return dxdt, dydt
