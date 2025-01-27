import streamlit as st

st.title("Practice App")
st.text(
    "The SIR model is a simple mathematical model to describe the spread of a disease in a population. It follows the following system of ODEs:"
)

# Create a column layout to align the equations to the left
col1, _ = st.columns(
    [1, 7]
)  # The first column (col1) is narrower to push content to the left

with col1:
    st.latex(
        r"""
    \frac{dS}{dt} = -\beta \cdot S \cdot I
    """
    )
    st.latex(
        r"""
    \frac{dI}{dt} = \beta \cdot S \cdot I - \gamma \cdot I
    """
    )
    st.latex(
        r"""
    \frac{dR}{dt} = \gamma \cdot I
    """
    )


st.markdown(
    r"""
    where $S$ is the susceptible population, $I$ is the infected population, 
    $R$ is the recovered population, $\beta$ is the transmission rate, 
    and $\gamma$ is the recovery rate. All quantities are adimensional. 
    
    
    Use the sliders below to change the transmission and recovery rates, 
    as well as the initial populations and the time interval to show the results.
    """
)

st.markdown("-------")

col1, col2 = st.columns(2)

with col1:
    transmission_rate = st.slider(
        "Transmission rate (beta)", min_value=0.00, max_value=1.00
    )
    recovery_rate = st.slider("Recovery rate (gamma)", min_value=0.00, max_value=1.00)

with col2:
    infected_pop = st.slider(
        "Initial Infected population (i0)", min_value=0.00, max_value=1.00
    )
    recovered_pop = st.slider(
        "Initial Recovered population (i0)", min_value=0.00, max_value=1.00
    )
    time = st.slider("Time (days)", min_value=1, max_value=200)

# transmission_rate = st.slider("Transmission rate (beta)", min_value=0.00, max_value=1.00)
# recovery_rate = st.slider("Recovery rate (gamma)", min_value=0.00, max_value=1.00)
# infected_pop = st.slider("Initial Infected population (i0)", min_value=0.00, max_value=1.00)
# recovered_pop = st.slider("Initial Recovered population (i0)", min_value=0.00, max_value=1.00)
# time = st.slider("Time (days)", min_value=1, max_value=200)

import numpy as np
from scipy.integrate import solve_ivp


def ode(t, y):
    s, i, r = y
    dsdt = -transmission_rate * s * i
    didt = transmission_rate * s * i - recovery_rate * i
    drdt = recovery_rate * i
    return [dsdt, didt, drdt]


t_span = [0, time]
# initial_cond = [0.95, 0.05, 0.0]
initial_cond = [1.0 - infected_pop - recovered_pop, infected_pop, recovered_pop]
sol = solve_ivp(ode, t_span, initial_cond)


import matplotlib.pyplot as plt

plt.grid(True)
plt.plot(sol.t, sol.y[0], color="blue", label="Susceptible")
plt.plot(sol.t, sol.y[1], color="orange", label="Infected")
plt.plot(sol.t, sol.y[2], color="green", label="Recovered")
plt.legend(["Susceptible", "Infected", "Recovered"])
plt.xlabel("Time")
plt.xlim(left=0)
plt.ylabel("Population")
plt.title("SIR model Disease Dynamics")
st.pyplot(plt)
