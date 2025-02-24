import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
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


# Initialize oscillators (phase and natural frequency)
# theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)


def kuramoto_order_parameter(theta: np.ndarray) -> tuple:
    # Compute the order parameter, r * exp(i * phi)
    order_param = np.mean(np.exp(1j * theta), axis=0)
    # The absolute value of the order parameter is the synchronization index
    r = np.abs(order_param)
    # The angle of the order parameter is the phase of the synchronization
    phi = np.angle(order_param)
    # The real part of the order parameter is r * cos(phi)
    rcosphi = np.real(order_param)
    # The imaginary part of the order parameter is r * sin(phi)
    rsinphi = np.imag(order_param)
    return r, phi, rcosphi, rsinphi


def kuramoto_critical_coupling(k: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute the theoretical order parameter for the Kuramoto model
    given the coupling strength k.

    Parameters
    ----------
    k : numpy.ndarray
        Coupling strength.
    sigma : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 1.0.

    Returns
    -------
    r : numpy.ndarray
        Order parameter.
    """
    # Compute the critical coupling strength kc
    kc = 2 * sigma * np.sqrt(2 / np.pi)
    r = np.zeros_like(k)
    r[k >= kc] = np.sqrt(1 - (kc / k[k >= kc]) ** 2)
    # Given an array of k, compute the theoretical order parameter
    # Remember its value differs for k < kc and k >= kc
    return r


def draw_kuramoto_diagram(
    num_oscillators: int = 100,
    sigma: float = 1.0,
    dt: float = 0.01,
    t_end: float = 100.0,
    kmin: float = 0.0,
    kmax: float = 5.0,
    knum: int = 7,
):
    """
    Draw the Kuramoto diagram, showing the order parameter as a function
    of the coupling strength. Theoretical and empirical order parameters
    are plotted.

    Parameters
    ----------
    num_oscillators : int, optional
        Number of oscillators, default is 1000.
    sigma : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 0.01.
    dt : float, optional
        Time step for the numerical integration, default is 0.01.
    t_end : float, optional
        End time for the numerical integration, default is 100.0.
    kmin : float, optional
        Minimum coupling strength, default is 0.0.
    kmax : float, optional
        Maximum coupling strength, default is 5.0.
    knum : int, optional
        Number of coupling strengths, default is 50.
    """
    # Time span and time points relevant for the numerical integration
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    # We will take the last X% of the time points to compute the order parameter
    idx_end = int(len(t_eval) * 0.25)
    t_eval = t_eval[-idx_end:]
    # Initialize the coupling strength and the empirical order parameter lists
    ls_k = np.linspace(kmin, kmax, knum)
    ls_r_q10 = np.zeros_like(ls_k)
    ls_r_q50 = np.zeros_like(ls_k)
    ls_r_q90 = np.zeros_like(ls_k)

    # Theoretical order parameter
    r_theoretical = kuramoto_critical_coupling(ls_k, sigma=sigma)

    # Initialize the oscillators
    theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)

    # Empirical order parameter
    for idx, coupling_strength in enumerate(ls_k):
        sol = solve_ivp(
            kuramoto_ode,
            t_span,
            theta,
            t_eval=t_eval,
            args=(omega, coupling_strength),
        )
        theta = sol.y
        # Keep theta within [0, 2 * pi]
        theta = np.mod(theta, 2 * np.pi)

        # Compute the order parameter
        r, phi, rcosphi, rsinphi = kuramoto_order_parameter(theta)

        # Append the mean order parameter of the last X% of the time points
        ls_r_q10[idx] = np.percentile(r, 10)
        ls_r_q50[idx] = np.percentile(r, 50)
        ls_r_q90[idx] = np.percentile(r, 90)

        print(
            f"K = {coupling_strength:.2f}, r (theory) = {r_theoretical[idx]:.2f}"
            f", r (empirical) = {ls_r_q50[idx]:.2f}"
        )

        # Take the last state as the initial condition for the next iteration
        theta = theta[:, -1]

    # Plot the order parameter as a function of time
    fig, ax = plt.subplots()
    ax.plot(ls_k, r_theoretical, label="Theoretical", color="blue")
    # Plot the empirical order parameter as points with error bars
    ax.errorbar(
        ls_k,
        ls_r_q50,
        yerr=[ls_r_q50 - ls_r_q10, ls_r_q90 - ls_r_q50],
        fmt="o",
        label="Empirical",
        color="red",
    )
    ax.set_xlabel("Coupling strength (K)")
    ax.set_ylabel("Order parameter (r)")
    ax.set_title("Kuramoto model")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    draw_kuramoto_diagram()
