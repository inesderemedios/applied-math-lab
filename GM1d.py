from typing import Tuple
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def gierer_meinhardt_pde(t, uv, a=0.4, d=20, gamma=1, b=1, dx=1):
    # compute the lapacian
    l = -2 * uv
    l = l + np.roll(uv, shift=1, axis=1)  # for the left
    l = l + np.roll(uv, shift=-1, axis=1)  # for the right
    l = l / dx**2
    u, v = uv
    lu, lv = l

    # odes
    f = a - b * u + u**2 / (v)
    g = u**2 - v
    dudt = lu + gamma * f  # task 1 says D1=1
    dvdt = d * lv + gamma * g  # task 1 says D2=d>0
    return np.array([dudt, dvdt])


def gm_jacobian(a=0.4, b=1):
    # Steps to compute the Jacobian:
    # 1) find fixed point of the system
    # 2) compute the derivatives at the fixed points
    fu = 2 * b / (a + 1) - b
    fv = -((b / (a + 1)) ** 2)
    gu = 2 * (a + 1) / b
    gv = -1.0
    return fu, fv, gu, gv


def is_turing_instability(a=0.40, b=1.00, d=30):
    fu, fv, gu, gv = gm_jacobian(a, b)
    # determinant of the Jacobian
    nabla = fu * gv - fv * gu
    # conditions
    c1 = (fu + gv) < 0  # trace
    c2 = nabla > 0  # determinant
    c3 = (gv + d * fu) > (2 * np.sqrt(d) * np.sqrt(nabla))
    return c1 & c2 & c3


def find_unstable_spatial_modes(
    a=0.40, b=1.00, d=30.0, length=40.0, num_modes=10, boundary_conditions="neumann"
):
    # jacobian matrix
    fu, fv, gu, gv = gm_jacobian(a, b)
    jac = np.array([[fu, fv], [gu, gv]])
    # check modes
    n_values = np.arange(num_modes)
    max_eigs = np.zeros(num_modes)

    for n in n_values:
        if boundary_conditions == "neumann":
            # For a 1D domain of length L with Neumann boundaries,
            # possible modes are k = n*pi/L, n = 0,1,2,...
            lambda_n = (n * np.pi / length) ** 2
        elif boundary_conditions == "periodic":
            lambda_n = ((n + 1) * np.pi / length) ** 2
        else:
            raise ValueError(
                "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
            )
        # eigenvalues of jacobian
        a_n = jac - lambda_n * np.diag([1, d])
        sigma1, sigma2 = np.linalg.eigvals(a_n)
        # only get the real aprt
        sigma1, sigma2 = sigma1.real, sigma2.real
        max_eigs[n] = max(sigma1, sigma2)

    # sort from largest to smallest eigenvalue
    sorted_indices = np.argsort(max_eigs)[::-1]
    # filter the modes that lead to Turing instability (positive eigenvalues)
    unstable_modes = sorted_indices[max_eigs[sorted_indices] > 0]
    return unstable_modes.tolist()


def run_simulation(
    gamma=1,
    b=1.00,
    dx=0.5,
    dt=0.001,
    anim_speed=100,
    length=40,
    seed=0,
    boundary_conditions="neumann",
):
    # variables modified by the user
    a = 0.40
    d = 20

    # Fix the random seed for reproducibility????
    np.random.seed(seed)

    # compute number of points in the 1D domain
    lenx = int(length / dx)

    # initialize the u and v fields
    uv = np.ones((2, lenx))
    # add 1% amplitude additive noise
    uv += uv * np.random.randn(2, lenx) / 100

    # initialize the x-axis
    x = np.linspace(0, length, lenx)

    # plot
    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax_ad: Axes = axs[0]  # a vs d
    ax_uv: Axes = axs[1]  # U vs V

    # This plane is to plot the Turing instability conditions
    # the function  designed works with arrays, so we will create a meshgrid to compute the Turing instability in the plane
    arr_a = np.linspace(0, 1, 1000)
    arr_d = np.linspace(0, 100, 1000)
    mesh_a, mesh_d = np.meshgrid(arr_a, arr_d)
    mask_turing = is_turing_instability(mesh_a, b, mesh_d)

    # plot the Turing instability region
    # contour plot to show the region where the conditions are met
    ax_ad.contourf(mesh_a, mesh_d, mask_turing, cmap="magma", alpha=0.5)

    # We also plot a point, that can be moved by the user
    (plot_adpoint,) = ax_ad.plot([a], [d], color="black", marker="o")

    ax_ad.grid(True)
    ax_ad.set_xlabel("a")
    ax_ad.set_ylabel("d")
    ax_ad.set_title("Turing Space")

    (plot_vline,) = ax_uv.plot(x, uv[1])

    # text to display the leading spatial modes
    plot_text = ax_uv.text(
        0.02 * length,
        4.95,
        "No Turing's instability",
        fontsize=12,
        verticalalignment="top",
    )
    plot_text2 = ax_uv.text(
        0.02 * length,
        4.7,
        "Click to change initial conditions",
        fontsize=12,
        verticalalignment="top",
    )

    # limits, title and labels; "legend"
    ax_uv.set_xlim(0, length)
    ax_uv.set_ylim(0, 5)
    ax_uv.set_xlabel("x")
    ax_uv.set_ylabel("v(x)")
    ax_uv.set_title("Gierer-Meinhardt Model (1D)")

    # This function will be called at each frame of the animation, updating the line objects

    def update_animation(frame: int, unstable_modes: list[int]):
        # Access the variables from the outer scope
        nonlocal a, d, uv

        # Iterate the simulation as many times as the animation speed
        # We use the Euler's method to integrate the ODEs
        # We cannot use solve_ivp because we must impose the boundary conditions
        # so at each iteration we do the follwoing:
        for _ in range(anim_speed):
            dudt = gierer_meinhardt_pde(0, uv, gamma=gamma, a=a, b=b, d=d, dx=dx)
            uv = uv + dudt * dt
            # The simulation may explode if the time step is too large
            # When this happens, we raise an error and stop the simulation
            if np.isnan(uv).any():
                raise ValueError("Simulation exploded. Reduce dt or increase dx.")
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[:, 0] = uv[:, 1]
                uv[:, -1] = uv[:, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )

        # Update the plot
        plot_vline.set_ydata(uv[1])
        plot_adpoint.set_data([a], [d])
        if len(unstable_modes) == 0:
            plot_text.set_text("No Turing's instability")
            plot_text2.set_text("")
        else:
            plot_text.set_text(f"Leading spatial mode: {unstable_modes[0]}")
            ls_modes = ", ".join(map(str, unstable_modes[1:8]))
            plot_text2.set_text(f"Unstable modes: {ls_modes}")

        return [plot_vline, plot_adpoint, plot_text, plot_text2]

    ani = animation.FuncAnimation(
        fig, update_animation, fargs=([],), interval=1, blit=True
    )

    # USER INTERACTION
    # We define a function that will be called when the user clicks on the graph
    # It will update the initial conditions and restart the animation

    def update_simulation(event: MouseEvent):
        # Access the a and d variables from the outer scope and modify them
        nonlocal a, d, uv

        # The click only works if it is inside the phase plane or stability diagram
        if event.inaxes == ax_ad:
            a = event.xdata
            d = event.ydata
        else:
            return

        # Reset to a constant line
        uv = np.ones((2, lenx)) * uv.mean()
        # Add 1% amplitude additive noise
        uv += uv * np.random.randn(2, lenx) / 100

        unstable_modes = find_unstable_spatial_modes(
            a=a,
            b=b,
            d=d,
            length=length,
            boundary_conditions=boundary_conditions,
        )

        # Stop the current animation, reset frame sequence and then start a new animation
        ani.event_source.stop()
        ani.frame_seq = ani.new_frame_seq()
        ani._args = (unstable_modes,)
        ani.event_source.start()

    # click event and update function are sinked
    fig.canvas.mpl_connect("button_press_event", update_simulation)

    # interactive plot
    plt.show()


if __name__ == "__main__":
    run_simulation()
