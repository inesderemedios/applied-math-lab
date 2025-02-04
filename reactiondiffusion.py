import numpy as np
import matplotlib.pyplot as plt

# Length of the rectangular region (L)
length = 40

# Initialize UV values as constant
uv = np.ones((2, 40))

# Add 1% noise to UV
uv += 0.01 * np.random.randn(*uv.shape)


def gierer_meinhard_1d(t, uv, a=0.4, d=20, gamma=1, b=1):
    u, v = uv
    # We have u(x) and v(x)
    # Assuming h = dx = 1, to get u(x + h)
    u_right = np.roll(u, shift=1)
    v_right = np.roll(v, shift=1)
    # Similarly for u(x - h), roll to opposite side
    u_left = np.roll(u, shift=-1)
    v_left = np.roll(v, shift=-1)
    # Laplacian via finite differences
    lu = u_right + u_left - 2 * u
    lv = v_right + v_left - 2 * v
    # Implement the rest of ODEs
    f = a - b * u + (u**2 / v)
    g = u**2 - v
    dudt = lu + gamma * f
    dvdt = d + gamma * g
    return np.array([dudt, dvdt])


num_iter = 50000
dt = 0.01

for iter in range(num_iter):
    # Solve ODE
    dudt, dvdt = gierer_meinhard_1d(num_iter, uv)
    # Update UV via Euler's method
    uv[0] = uv[0] + dudt * dt  # u
    uv[1] = uv[1] + dvdt * dt  # v

    # Neumann boundary conditions
    uv[:, 0] = uv[:, 1]
    uv[:, -1] = uv[:, -2]


# Initialize the figure and axes
fig, ax = plt.subplots(nrows=1, ncols=1)

# Initialize the x
x = np.linspace(0, length, uv.shape[1])

# Plot v(x)
ax.plot(x, uv[1])

# Add some elements to make figure prettier
# ax.set_ylim(0, 2)
ax.set_xlabel("x")
ax.set_ylabel("v(x)")
ax.legend(["v(x)"])

plt.plot(np.linspace(0, length, uv.shape[1]), uv[1], label="v(x)")
plt.xlabel("x")
plt.ylabel("v(x)")
plt.legend()
plt.show()
