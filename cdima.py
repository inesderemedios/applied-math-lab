import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
import matplotlib.pyplot as plt


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


def find_root(a, b):
    def equations(vars):
        x, y = vars
        dxdt, dydt = cdima(a, b, x, y)
        return [dxdt, dydt]

    root = fsolve(equations, (0, 0))
    return root


def trajectory(a, b, x0, y0, t_span, t_eval):
    solution = rk4_cdima(a, b, x0, y0, t_span, t_eval)
    return solution.t, solution.y[0], solution.y[1]


def nullclines(a, b, x_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    # dxdt=0
    nullcline1 = (a - x) - ((1 + x**2) / (4 * x))
    # dydt=0
    nullcline2 = (1 + x**2) * np.ones_like(
        x
    )  # x=0 is trivial, bu then for any value of x, the value of y on the nullcline is 1+ x**2
    return x, nullcline1, nullcline2


a = 10
b = 6
t_span = (0, 100)
t_eval = np.linspace(0, 10, 100)
x0, y0 = 0, 0
t, x, y = trajectory(a, b, x0, y0, t_span, t_eval)
x_range = (0, 5)
solution = rk4_cdima(a, b, x0, y0, t_span, t_eval)
r = find_root(a, b)
root = tuple(r)
print(solution)
print(root)
