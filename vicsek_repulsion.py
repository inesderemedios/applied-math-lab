import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def initialize_particles(
    num_boids: int, box_size: float = 25
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles.

    Parameters
    ----------
    num_boids : int
        Number of particles.
    box_size : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial positions of the particles.
    np.ndarray
        Initial angle of the particles, in radians.
    """
    # REPLACE THE TWO LINES BELOW
    # Random initial theta
    theta = np.random.uniform(0, 2 * np.pi, num_boids)
    # Random initial x, y
    xy = np.random.uniform(0, box_size, (2, num_boids))
    return xy, theta


def vicsek_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    box_size: float = 25,
    radius_interaction: float = 1,
    v0: float = 0.03,
    repulsion: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    num_boids = xy.shape[1]
    new_xy = xy.copy()
    """
    Update the state of the particles based on the Vicsek model.

    Parameters
    ----------
    xy : np.ndarray
        Position of the particles.
    theta : np.ndarray
        Angle of the particles.
    dt : float, optional
        Time step, default is 1 (standard convention).
    eta : float, optional
        Noise parameter, default is 0.1.
    box_size : float, optional
        Dimension of the space, default is 25.
    radius_interaction : float, optional
        Interaction radius, default is 1 (standard convention).
    v0 : float, optional
        Speed of the particles, default is 0.03.
    repulsion : float, optional
        Repulsion parameter, default is 0.1.

    Returns
    -------
    np.ndarray
        Updated position of the particles.
    np.ndarray
        Updated angle of the particles.
    """
    for i in range(num_boids):
        # Compute the distance between the particles
        distances = np.linalg.norm(xy - xy[:, i : i + 1], axis=0)

        # apply the repulsion if its in the interation radius
        for j in range(num_boids):
            if i != j and distances[j] < radius_interaction:
                # calculate the repulsion vector:
                repulsion_vector = xy[:, i] - xy[:, j]
                # normalize the vector
                repulsion_vector = repulsion_vector / np.linalg.norm(repulsion_vector)
                # add the repulsion vector to the position
                new_xy[:, i] += repulsion_vector * repulsion

    # update the position
    theta = np.mean(theta) + eta * np.random.uniform(-np.pi, np.pi, len(theta))
    new_xy = xy + v0 * np.array([np.cos(theta), np.sin(theta)]) * dt
    new_xy = np.mod(xy, box_size)
    return new_xy + 0.01, theta + 0.01


def vicsek_order_parameter(theta: np.ndarray) -> float:
    """
    Compute the order parameter of the Vicsek model.

    Parameters
    ----------
    theta : np.ndarray
        Angle of the particles.

    Returns
    -------
    float
        Order parameter of the Vicsek model.
    """
    order_param = np.abs(np.mean(np.exp(1j * theta)))
    return order_param


def run_simulation(dt: float = 1):
    """
    Run the animation of the Vicsek model.
    Reference: Topaz, Chad M., Lori Ziegelmeier, and Tom Halverson. 2015.
    "Topological Data Analysis of Biological Aggregation Models."
    PLOS ONE 10 (5): e0126383. https://doi.org/10/f7mp7k

    Parameters
    ----------
    dt : float, optional
        Time step, default is 1 (standard convention).
    """
    # Initialize parameters (will be changed with sliders)
    num_boids = 300
    noise_eta = 0.05
    box_size = 25
    radius_interaction = 1
    v0 = 0.03
    repulsion_strength = 0.1

    # Initialize particles
    xy, theta = initialize_particles(num_boids, box_size=box_size)
    ls_order_param = [0] * 3000
    dict_noise = {}

    # Plot particles to the left, order parameter to the right
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[4, 1])
    ax_plane: Axes = axs[0, 0]
    ax_order: Axes = axs[0, 1]
    ax_sliders: Axes = axs[1, 0]
    ax_noise: Axes = axs[1, 1]

    # Initialize quiver plot
    plt_particles = ax_plane.quiver(
        xy[0],
        xy[1],
        np.cos(theta),
        np.sin(theta),
        angles="xy",
    )
    ax_plane.set_xlim(0, box_size)
    ax_plane.set_ylim(0, box_size)
    ax_plane.set_aspect("equal")

    # Initialize order parameter
    (line_order_param,) = ax_order.plot([], [])
    ax_order.set_xlim(0, 3000)
    ax_order.set_ylim(0, 1)
    ax_order.set_xlabel("Time")
    ax_order.set_ylabel("Order parameter (r)")
    ax_order.grid(True)

    # Initialize order parameter vs noise
    (line_noise,) = ax_noise.plot([], [], color="red", marker="o", linestyle="--")
    ax_noise.set_xlim(0, 1)
    ax_noise.set_ylim(0, 1)
    ax_noise.set_xlabel("Noise (eta)")
    ax_noise.set_ylabel("Order param (r)")

    # Clear axis for the sliders
    ax_sliders.axis("off")

    # --------------------------------
    # ANIMATION
    # --------------------------------

    def update_animation(frame: int):
        nonlocal xy, theta, noise_eta, v0, radius_interaction, box_size, dict_noise
        xy, theta = vicsek_equations(
            xy,
            theta,
            v0=v0,
            dt=dt,
            radius_interaction=radius_interaction,
            box_size=box_size,
            eta=noise_eta,
        )

        # Update quiver plot
        plt_particles.set_offsets(xy.T)
        plt_particles.set_UVC(np.cos(theta), np.sin(theta))

        # Update order parameter
        ls_order_param.append(vicsek_order_parameter(theta))
        ls_order_param.pop(0)
        line_order_param.set_data(range(3000), ls_order_param)

        # Average the last 1000 values to get the order parameter
        order_param = np.mean(ls_order_param[-1000:])
        dict_noise[noise_eta] = order_param
        dict_noise = dict(sorted(dict_noise.items()))
        line_noise.set_data(*zip(*dict_noise.items()))
        return plt_particles, line_order_param, line_noise

    ani = animation.FuncAnimation(fig, update_animation, interval=0, blit=True)

    # --------------------------------
    # SLIDERS
    # --------------------------------

    # Add sliders
    ax_num_boids = ax_sliders.inset_axes([0.0, 1.2, 0.8, 0.1])
    ax_radius_interaction = ax_sliders.inset_axes([0.0, 1.0, 0.8, 0.1])
    ax_noise_eta = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    ax_v0 = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    ax_box_size = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])
    ax_repulsion = ax_sliders.inset_axes([0.0, 0.2, 0.8, 0.1])

    slider_num_boids = plt.Slider(
        ax_num_boids, "Number of boids", 100, 1000, valinit=num_boids, valstep=100
    )
    slider_radius_interaction = plt.Slider(
        ax_radius_interaction,
        "Interaction radius",
        0,
        50,
        valinit=radius_interaction,
        valstep=1,
    )
    slider_noise_eta = plt.Slider(
        ax_noise_eta, "Noise", 0.0, 0.5, valinit=noise_eta, valstep=0.01
    )
    slider_v0 = plt.Slider(ax_v0, "Speed", 0.0, 0.1, valinit=v0, valstep=0.01)
    slider_box_size = plt.Slider(
        ax_box_size, "Dimension", 1, 50, valinit=box_size, valstep=1
    )

    slider_repulsion = plt.Slider(
        ax_repulsion, "Repulsion", 0.0, 1.0, valinit=repulsion_strength, valstep=0.01
    )

    def update_sliders(_):
        nonlocal xy, radius_interaction, noise_eta, v0, box_size
        # Pause animation
        ani.event_source.stop()

        # Update parameters with sliders
        noise_eta = slider_noise_eta.val
        v0 = slider_v0.val
        box_size = slider_box_size.val
        radius_interaction = slider_radius_interaction.val
        repulsion_strength = slider_repulsion.val

        # Update plot limits
        ax_plane.set_xlim(0, box_size)
        ax_plane.set_ylim(0, box_size)
        ax_plane.set_aspect("equal")

        # Reinitialize the animation
        ani.event_source.start()

    slider_radius_interaction.on_changed(update_sliders)
    slider_noise_eta.on_changed(update_sliders)
    slider_v0.on_changed(update_sliders)
    slider_box_size.on_changed(update_sliders)
    slider_repulsion.on_changed(update_sliders)

    # A special case must be done for the number of boids as it requires to reinitialize the particles
    def update_num_boids(_):
        nonlocal xy, theta, num_boids, plt_particles
        # Pause animation
        ani.event_source.stop()

        # Update number of boids
        num_boids = int(slider_num_boids.val)
        # Reinitialize particles
        xy, theta = initialize_particles(num_boids, box_size=box_size)

        # Because the number of particles has changed, we need to redefine the quiver plot
        # This is because the number of particles on the plot is fixed at the beginning
        # and cannot be changed dynamically
        plt_particles.remove()  # Remove old quiver plot
        plt_particles = ax_plane.quiver(
            xy[0],
            xy[1],
            np.cos(theta),
            np.sin(theta),
            angles="xy",
        )

        # Reinitialize the animation
        ani.event_source.start()

    slider_num_boids.on_changed(update_num_boids)

    # --------------------------------
    # SHOW
    # --------------------------------

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
