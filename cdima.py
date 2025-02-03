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


a = 8
b = 3
t_span = (0, 100)
t_eval = np.linspace(0, 10, 1000)
x0, y0 = 0, 0
t, x, y = trajectory(a, b, x0, y0, t_span, t_eval)
x_range = (0, 5)
solution = rk4_cdima(a, b, x0, y0, t_span, t_eval)
r = find_root(a, b)
root = tuple(r)
print(solution)
print(root)

# animation
fig, ax = plt.subplots()
(line,) = ax.plot([], [], "b-")
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_xlabel("x")
ax.set_ylabel("y")


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    line.set_data(x[:frame], y[:frame])
    return (line,)


ani = animation.FuncAnimation(
    fig, update, frames=len(t), init_func=init, blit=True, interval=1
)
# Plot fixed point
fixed_point = ax.plot(root[0], root[1], "ro", label="Fixed Point")

# Plot nullclines
x_null, nullcline1, nullcline2 = nullclines(a, b, x_range)
nullcline1_line = ax.plot(x_null, nullcline1, "g--", label="dx/dt=0 Nullcline")
nullcline2_line = ax.plot(x_null, nullcline2, "m--", label="dy/dt=0 Nullcline")

# legend
ax.legend()
plt.show()
