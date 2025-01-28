import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


def cdima(a, b, x, y):
    dxdt = a - x - ((4 * x * y) / (1 + x**2))
    dydt = b * x * (1 - (y / (1 + x**2)))
    return dxdt, dydt


def rk4_cdima(a, b, x0, y0, t_span, t_eval):
    def integral(t, vars):
        x, y = vars
        dxdt, dydt = cdima(a, b, x, y)
        return [dxdt, dydt]

    sol = solve_ivp(integral, t_span, [x0, y0], t_eval=t_eval)
    return sol


a = 10
b = 6
t_span = (0, 100)
t_eval = np.linspace(0, 10, 100)
x0, y0 = 0, 0
solution = rk4_cdima(a, b, x0, y0, t_span, t_eval)
print(solution)


def find_root(a, b):
    def equations(vars):
        x, y = vars
        dxdt, dydt = cdima(a, b, x, y)
        return [dxdt, dydt]

    root = fsolve(equations, (0, 0))
    return root


a = 1.0
b = 2.0
r = find_root(a, b)
root = tuple(r)
print(root)
