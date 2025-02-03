import numpy as np
import scipy as sp
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def vanderpol(t, vars, mu):
    x, y = vars
    fx = (x**3 / 3) - x
    dx_dt = mu * (y - fx)
    dy_dt = -x / mu
    return np.array([dx_dt, dy_dt])


def rk4_vanderpol(x0, y0, mu, t_span, t_eval):
    sol = solve_ivp(vanderpol, t_span, [x0, y0], args=(mu,), t_eval=t_eval)
    return sol


def find_root(mu):
    def equations(vars):
        x, y = vars
        dxdt, dydt = vanderpol(0, [x, y], mu)
        return [dxdt, dydt]

    root = fsolve(equations, (0, 0))
    return root


def trajectory(x0, y0, mu, t_span, t_eval):
    sol = rk4_vanderpol(x0, y0, mu, t_span, t_eval)
    return sol.t, sol.y[0], sol.y[1]


def nullclines(mu, x_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    fx = (x**3 / 3) - x
    nullcline1 = fx
    nullcline2 = -x / mu
    return x, nullcline1, nullcline2


mu = 1
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
x0, y0 = 5, 1
t, x, y = trajectory(x0, y0, mu, t_span, t_eval)
x_range = (-2, 2)
r = find_root(mu)
root = tuple(r)

# animation
fig, ax = plt.subplots()
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_xlabel("x")
ax.set_ylabel("y")

# nullclines
x_null, nullcline1, nullcline2 = nullclines(mu, x_range)
ax.plot(x_null, nullcline1, "g--", label="dx/dt=0 Nullcline")
ax.plot(x_null, nullcline2, "m--", label="dy/dt=0 Nullcline")

# fixed point
ax.plot(root[0], root[1], "ro", label="Fixed Point")

# plot f(x)
F_x = (1 / 3 * x**3) - x
ax.plot(x, F_x, "k-", label="F(x)")
ax.legend()

# trajectory
(line,) = ax.plot([], [], "b-", alpha=0.8)


def init():
    line.set_data([], [])
    return (line,)


def update(frame):
    line.set_data(x[:frame], y[:frame])
    return (line,)


ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=False, interval=10)

plt.legend()
plt.grid()
plt.show()
