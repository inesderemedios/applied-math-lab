import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.integrate import solve_ivp


def initialize_oscillators(num_oscillators: int, sigma: float = 1.0):
    # Assign a random initial phase to each oscillator (position in the unit circle)
    theta = np.random.uniform(0, 2 * np.pi, num_oscillators)
    # Assign a random natural frequency to each oscillator (angular velocity)
    omega = np.random.normal(0, sigma, num_oscillators)
    return theta, omega


def kuramoto_ode(
    t: float, theta: np.ndarray, omega: np.ndarray = 1, coupling_strength: float = 1.0
) -> np.ndarray:
    # Keep theta within [0, 2 * pi]
    theta = np.mod(theta, 2 * np.pi)

    # Your ODE goes here
    dtheta_dt = omega + (coupling_strength) * np.mean(
        np.sin(theta[:, None] - theta), axis=0
    )

    return dtheta_dt


# --------------------------------------------#
# PARAMETERS
# --------------------------------------------#

coupling_strength = 1.0  # K
num_oscillators = 100
sigma = 1.0
dt = 0.01

# Initialize oscillators (phase and natural frequency)
theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)

# --------------------------------------------#
# FIGURE SETUP
# --------------------------------------------#

fig, ax_phase = plt.subplots(1, 1, figsize=(12, 6))

ax_phase.set_title("Kuramoto Model")
ax_phase.set_xlabel("Cos(theta)")
ax_phase.set_ylabel("Sin(theta)")
ax_phase.set_xlim(-1.1, 1.1)
ax_phase.set_ylim(-1.1, 1.1)
ax_phase.set_aspect("equal")
ax_phase.grid(True)

# Draw unit circle
circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
ax_phase.add_artist(circle)

# Initialize scatter plot for oscillators
scatter = ax_phase.scatter([], [], s=50, color="blue", alpha=0.5)

# --------------------------------------------#
# ANIMATION
# --------------------------------------------#


def update(frame: int):
    # Acces the variables from the outer scope
    global theta

    # Solve the ODE system
    sol = solve_ivp(
        kuramoto_ode,
        (0, dt),
        theta,
        args=(omega, coupling_strength),
    )
    # Solution are a 2D array with shape (N, T)
    theta = sol.y[..., -1]

    # Keep theta within [0, 2 * pi]
    theta = np.mod(theta, 2 * np.pi)

    # Update scatter plot on the unit circle
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.vstack((x, y)).T
    scatter.set_offsets(data)
    return [scatter]


ani = animation.FuncAnimation(fig, update, blit=True, interval=1)

# --------------------------------------------#
# RUN AND SHOW
# --------------------------------------------#

plt.tight_layout()
plt.show()
