import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random
from matplotlib.widgets import Slider


edges = pd.read_csv(
    r"C:\Users\localadmin\Downloads\applied math lab\applied-math-lab-1\facebook_combined.txt",
    sep=" ",
    names=["source", "target"],
)

G = nx.from_pandas_edgelist(edges)

# Initialize node states: "I"gnorant (blue), "S"preader (red), "R" Stifler (green)
for node in G.nodes:
    G.nodes[node]["state"] = "I"  # Ignorant
initial_spreader = random.choice(list(G.nodes))
G.nodes[initial_spreader]["state"] = "S"  # Start with one spreader

# Layout for consistent node positions
pos = nx.spring_layout(G, seed=42)

# Initialize SIR counts
sir_data = {"I": [], "S": [], "R": []}

# Parameters (will be controlled with sliders)
beta = 0.3
gamma = 0.1

# Matplotlib figure and axes
fig, (ax_graph, ax_curve) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25)

# Sliders
ax_beta = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_gamma = plt.axes([0.15, 0.05, 0.65, 0.03])
slider_beta = Slider(ax_beta, "Beta (Spread)", 0.0, 1.0, valinit=beta)
slider_gamma = Slider(ax_gamma, "Gamma (Stifle)", 0.0, 1.0, valinit=gamma)

fig.text(
    0.15,
    0.01,
    "Adjust sliders to change infection and recovery rates in real time",
    fontsize=9,
)


# Update node states using SIR model
def update_states():
    global G
    to_spread = []
    to_stifle = []
    for node in G.nodes:
        if G.nodes[node]["state"] == "S":
            for neighbor in G.neighbors(node):
                if (
                    G.nodes[neighbor]["state"] == "I"
                    and random.random() < slider_beta.val
                ):
                    to_spread.append(neighbor)
            if random.random() < slider_gamma.val:
                to_stifle.append(node)
    for node in to_spread:
        if G.nodes[node]["state"] == "I":
            G.nodes[node]["state"] = "S"
    for node in to_stifle:
        G.nodes[node]["state"] = "R"


# Plot functions
def plot_graph():
    ax_graph.clear()
    color_map = {"I": "blue", "S": "red", "R": "green"}
    node_colors = [color_map[G.nodes[n]["state"]] for n in G.nodes]
    nx.draw(
        G, pos, ax=ax_graph, node_color=node_colors, node_size=30, with_labels=False
    )
    ax_graph.set_title("Facebook Network")


def plot_sir():
    ax_curve.clear()
    I = sum(1 for n in G.nodes if G.nodes[n]["state"] == "I")
    S = sum(1 for n in G.nodes if G.nodes[n]["state"] == "S")
    R = sum(1 for n in G.nodes if G.nodes[n]["state"] == "R")
    sir_data["I"].append(I)
    sir_data["S"].append(S)
    sir_data["R"].append(R)
    ax_curve.plot(sir_data["I"], label="Ignorant", color="blue")
    ax_curve.plot(sir_data["S"], label="Spreader", color="red")
    ax_curve.plot(sir_data["R"], label="Stifler", color="green")
    ax_curve.set_title("SIR Curve")
    ax_curve.set_xlabel("Time")
    ax_curve.set_ylabel("Count")
    ax_curve.legend()


# Main animation loop
for step in range(100):
    update_states()
    plot_graph()
    plot_sir()
    plt.pause(0.2)

plt.show()
