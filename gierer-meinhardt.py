import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Length of the rectangular region (L)
length_x = 20
length_y = 50

# Initialize UV values as constant
uv = np.ones((2, 20, 50))

# Add 1% noise to UV
uv = uv + uv * np.random.uniform(0, 1, uv.shape) / 100


def gierer_meinhard_2d(t, uv, a=0.4, d=40, gamma=1, b=1):
    u, v = uv
    # We have u(x) and v(x)
    # Assuming h = dx = 1, to get u(x + h)
    u_up = np.roll(u, shift=1, axis=0)
    v_up = np.roll(v, shift=1, axis=0)
    # Similarly for u(x - h), roll to opposite side
    u_down = np.roll(u, shift=-1, axis=0)
    v_down = np.roll(v, shift=-1, axis=0)
    #  Moving left
    u_left = np.roll(u, shift=1, axis=1)
    v_left = np.roll(v, shift=1, axis=1)
    #  Moving right
    u_right = np.roll(u, shift=-1, axis=1)
    v_right = np.roll(v, shift=-1, axis=1)
    # lapacian
    lu = u_right + u_left + u_up + u_down - 4 * u  # divided by h^2 = 1
    lv = v_right + v_left + v_up + v_down - 4 * v  # divided by h^2 = 1
    # implement the rest of pdes
    f = a - b * u + (u**2 / v)
    g = u**2 - v
    dudt = lu + gamma * f
    dvdt = d * lv + gamma * g
    return np.array([dudt, dvdt])


print(uv.shape)

# Create a figure and axis for plotting
fig, ax_uv = plt.subplots()

# Initialize the image object for animation
im = ax_uv.imshow(
    uv[1],
    interpolation="bilinear",
    origin="lower",
    extent=[0, length_y, 0, length_x],
)


def animate(frame: int):
    # Solve PDE
    num_iter = 50000
    dt = 0.0001
    dudt, dvdt = gierer_meinhard_2d(num_iter, uv)
    # Update UV
    # Update UV via Euler's method
    uv[0] = uv[0] + dudt * dt  # u
    uv[1] = uv[1] + dvdt * dt

    # Neumann boundary conditions in 2D
    uv[:, 0, :] = uv[:, 1, :]
    uv[:, -1, :] = uv[:, -2, :]
    uv[:, :, 0] = uv[:, :, 1]
    uv[:, :, -1] = uv[:, :, -2]
    im.set_array(uv[1])
    return (im,)


# im.set_clim(vmin=uv[1].min(), vmax=uv[1].max() + 0.1)

ani = animation.FuncAnimation(
    fig,
    animate,
    interval=1,
    blit=True,
)

plt.show()
