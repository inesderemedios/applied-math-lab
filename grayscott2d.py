import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.widgets import Slider


def laplacian(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 5-point finite
    difference scheme: considering each point and its immediate neighbors in
    the up, down, left, and right directions.

    Reference: https://en.wikipedia.org/wiki/Five-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian
        of u and v.
    """
    lap = -4 * uv

    # Immediate neighbors (up, down, left, right)
    lap += np.roll(uv, shift=1, axis=1)  # up
    lap += np.roll(uv, shift=-1, axis=1)  # down
    lap += np.roll(uv, shift=1, axis=2)  # left
    lap += np.roll(uv, shift=-1, axis=2)  # right
    return lap


def laplacian_9pt(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 9-point finite
    difference scheme (Patra-Karttunen), considering each point and its
    immediate neighbors, including diagonals.

    Reference: https://en.wikipedia.org/wiki/Nine-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of u and v.

    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian of u and v.
    """
    # Weights for the 9-point stencil (Patra-Karttunen)
    center_weight = -20 / 6
    neighbor_weight = 4 / 6
    diagonal_weight = 1 / 6

    lap = center_weight * uv

    # Shifted arrays for immediate neighbors
    up = np.roll(uv, shift=1, axis=1)
    down = np.roll(uv, shift=-1, axis=1)

    # Immediate neighbors (up, down, left, right)
    lap += neighbor_weight * up  # up
    lap += neighbor_weight * down  # down
    lap += neighbor_weight * np.roll(uv, shift=1, axis=2)  # left
    lap += neighbor_weight * np.roll(uv, shift=-1, axis=2)  # right

    # Diagonal neighbors
    lap += diagonal_weight * np.roll(up, shift=1, axis=2)  # up-left
    lap += diagonal_weight * np.roll(up, shift=-1, axis=2)  # up-right
    lap += diagonal_weight * np.roll(down, shift=1, axis=2)  # down-left
    lap += diagonal_weight * np.roll(down, shift=-1, axis=2)  # down-right

    return lap


def gray_scott_pde(
    t: float,
    uv: np.ndarray,
    d1: float = 0.1,
    d2: float = 0.05,
    f: float = 0.040,
    k: float = 0.060,
    stencil: int = 5,
) -> np.ndarray:
    """
    Update the u and v fields using the Gray-Scott model with explicit Euler
    time integration, where the fields are updated based on their current values
    and the calculated derivatives.

    Reference: https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    d1 : float, optional
        Diffusion rate of u, default is 0.1.
    d2 : float, optional
        Diffusion rate of v, default is 0.05.
    f : float, optional
        Feed rate (at which u is fed into the system), default is 0.040.
    k : float, optional
        Kill rate (at which v is removed from the system), default is 0.060.
    stencil : int, optional
        Stencil to use for the Laplacian computation. Use 5 or 9, default is 5.
    """

    # Extract the matrices for substances u and v
    u, v = uv

    # Compute the Laplacian of u and v
    if stencil == 5:
        lap = laplacian(uv)
    elif stencil == 9:
        lap = laplacian_9pt(uv)
    else:
        raise ValueError("Invalid stencil value. Use 5 or 9.")

    # Extract the Laplacian matrices for u and v
    lu, lv = lap

    uv2 = u * v * v

    # Gray-Scott equations
    du_dt = d1 * lu - uv2 + f * (1 - u)
    dv_dt = d2 * lv + uv2 - (f + k) * v

    return np.array([du_dt, dv_dt])


def run_simulation(
    n: int = 250,
    dx: float = 1,
    dt: float = 2,
    anim_speed: int = 100,
    cmap: str = "jet",
):
    """
    Animate the Gray-Scott model simulation.

    Parameters
    ----------
    n : int
        Number of grid points in one dimension, N.
    dx : float
        Spacing between grid points.
    dt : float
        Time step.
    boundary_conditions : str
        Boundary conditions to apply. Use 'neumann' or 'periodic'.
    anim_speed : int
        Animation speed. Number of iterations per frame.
    cmap : str
        Colormap to use for the plot, by default 'jet'.
    """
    # ------------------------------------------------------------------------#
    # PARAMETERS
    # ------------------------------------------------------------------------#
    length = n * dx  # L

    # Initial parameters - Will be changed using the sliders
    d1 = 0.1
    d2 = 0.05
    f = 0.082
    k = 0.059
    boundary_conditions = "periodic"

    pause = False
    drawing = False
    # Initialize the (u, v) = (1, 0)
    uv = np.ones((2, length, length), dtype=np.float32)
    uv[1] = 0

    # Create figure with plot on the left (6x6) and sliders on the right (6x4)
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 6), gridspec_kw={"width_ratios": [6, 4]}
    )

    # Initialize the v field image
    ax_uv: Axes = axs[0]
    ax_uv.axis("off")  # Turn off the axis (the grid and numbers)
    im = ax_uv.imshow(uv[1], cmap=cmap, interpolation="bilinear", vmin=0, vmax=1.0)

    def update_frame(_):
        # Access variables from the outer scope
        nonlocal pause, uv, d1, d2, f, k, boundary_conditions
        if pause:
            return [im]

        for _ in range(anim_speed):
            # Solve an initial value problem for a system of ODEs via Euler's method
            uv = uv + gray_scott_pde(_, uv, d1=d1, d2=d2, f=f, k=k) * dt
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[:, 0, :] = uv[:, 1, :]
                uv[:, -1, :] = uv[:, -2, :]
                uv[:, :, 0] = uv[:, :, 1]
                uv[:, :, -1] = uv[:, :, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )

        im.set_array(uv[1])
        return [im]  # Elements to update (using blit=True)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_frame, interval=1, blit=True)

    # ------------------------------------------------------------------------#
    # ANIMATION - PAUSE / RESUME
    # ------------------------------------------------------------------------#

    # We want the user to be able to pause or resume the simulation by pressing the space bar
    # In order to do this, we need a key press event handler: on_keypress

    def on_keypress(event: KeyEvent):
        """This function is called when the user presses a key.
        It pauses or resumes the simulation when the space bar is pressed."""
        # Pressing the space bar pauses or resumes the simulation
        if event.key == " ":
            nonlocal pause
            pause ^= True

    # Attach the key press event handler to the figure
    fig.canvas.mpl_connect("key_press_event", on_keypress)

    # ------------------------------------------------------------------------#
    # SLIDERS
    # ------------------------------------------------------------------------#

    # The following sliders will allow the user to change the parameters of the Gray-Scott model

    # Create the sliders axes
    ax_sliders: Axes = axs[1]
    ax_sliders.axis("off")  # Turn off the axis (the grid and numbers)

    # Place the axes objects that will contain the sliders
    # We define the location of each axes inside the right column of the figure
    ax_d1 = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])  # [x0, y0, width, height]
    ax_d2 = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])  # [x0, y0, width, height]
    ax_f = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])  # [x0, y0, width, height]
    ax_k = ax_sliders.inset_axes([0.0, 0.2, 0.8, 0.1])  # [x0, y0, width, height]
    ax_bc = ax_sliders.inset_axes([0.3, 0.05, 0.2, 0.1])  # [x0, y0, width, height]

    # Create the sliders, each in its own axes [min, max, initial]
    slider_d1 = Slider(ax_d1, "D1", 0.01, 0.2, valinit=d1, valstep=0.01)
    slider_d2 = Slider(ax_d2, "D2", 0.01, 0.2, valinit=d2, valstep=0.01)
    slider_f = Slider(ax_f, "F", 0.01, 0.09, valinit=f, valstep=0.001)
    slider_k = Slider(ax_k, "k", 0.04, 0.07, valinit=k, valstep=0.001)
    slider_bc = Slider(
        ax_bc, boundary_conditions.capitalize(), 0, 1, valinit=0, valstep=1
    )

    # This function will be called when the user changes the sliders
    def update_sliders(_):
        # Acces the variables from the outer scope to update them
        nonlocal d1, d2, f, k, boundary_conditions, pause
        # Update the parameters according to the sliders values
        d1 = slider_d1.val
        d2 = slider_d2.val
        f = slider_f.val
        k = slider_k.val
        boundary_conditions = "periodic" if slider_bc.val == 0 else "neumann"
        # Change slider text
        slider_bc.label.set_text(boundary_conditions.capitalize())
        # Pause the simulation when sliders are updated
        pause = True

    # Attach the update function to sliders
    slider_d1.on_changed(update_sliders)
    slider_d2.on_changed(update_sliders)
    slider_f.on_changed(update_sliders)
    slider_k.on_changed(update_sliders)
    slider_bc.on_changed(update_sliders)

    # ------------------------------------------------------------------------#
    #  DRAW ON UV PLOT
    # ------------------------------------------------------------------------#

    # We want the user to be able to add and remove sources of v by clicking on the plot
    # The first function we define is update_uv, which will be called when the user clicks on the plot
    # It will update the u and v fields based on the mouse position and the mouse button pressed
    # Right click will remove the source of v, left click will add a source of v

    def update_uv(event: MouseEvent, r: int = 3):
        if event.inaxes != ax_uv:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)

        # Left click? Add a perturbation (u, v) = (0.5, 0.5)
        # plus an additive noise of 10% that value
        if event.button == 1:
            u_new = 0.5 * (1 + 0.1 * np.random.randn())
            v_new = 0.5 * (1 + 0.1 * np.random.randn())
        # Right click? Reset to initial values (u, v) = (1, 0)
        elif event.button == 3:
            u_new = 1.0
            v_new = 0.0
        else:
            return

        uv[0, y - r : y + r, x - r : x + r] = u_new
        uv[1, y - r : y + r, x - r : x + r] = v_new

        # Update the displayed image
        im.set_array(uv[1])

    # Next, we want the user to be able to draw lines on the plot to add or remove sources of v
    # In order to do this, we need to define some mouse event handlers: on_click, on_release, on_motion

    def on_click(event: MouseEvent):
        """This function is called when the user clicks on the plot.
        It initializes the drawing process."""
        if event.inaxes != ax_uv:
            return
        if event.xdata is None or event.ydata is None:
            return
        nonlocal drawing
        drawing = True
        update_uv(event)

    def on_release(event: MouseEvent):
        """This function is called when the user releases the mouse button.
        It stops the drawing process."""
        nonlocal drawing
        drawing = False

    def on_motion(event: MouseEvent):
        """This function is called when the user moves the mouse.
        It updates the u and v fields if the drawing process is active.
        """
        if drawing:
            update_uv(event)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # ------------------------------------------------------------------------#
    #  DISPLAY
    # ------------------------------------------------------------------------#

    plt.show()


if __name__ == "__main__":
    run_simulation()
