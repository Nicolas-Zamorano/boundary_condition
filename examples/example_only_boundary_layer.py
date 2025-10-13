"""Example of approximating the boundary condition of a function using a NN."""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import (
    FeedForwardNeuralNetwork as FNN,
    C0BoundaryModel,
    C2BoundaryModel,
    Integration,
)

torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float64)

## ---- BOUNDARY CONDITION ---- ##

# NN = C0BoundaryModel()
NN = C2BoundaryModel()

optimizer = torch.optim.Adam(NN.parameters(), lr=1e-1)

### ---- LOSS PARAMETERS ---- ####

EPSILON = 2e-2
SCALING_FACTOR = 1.1


def f(x: torch.Tensor) -> torch.Tensor:
    """Function to approximate his boundary condition."""
    return SCALING_FACTOR * (1 - torch.exp(-x / EPSILON)) * (1 - x)


def loss_function(points: torch.Tensor, value_f: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return (NN(points) - value_f) ** 2


integral_rule = Integration(
    intervals_start=0.5, interval_end=1, nb_intervals=100, integration_order=2
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 15000

training_bar = tqdm(range(EPOCHS))

training_history = []

best_loss = float("inf")
MIN_DELTA = 1e-16
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_COUNTER = 0

f_value = f(integral_rule.integration_points)

### ---- TRAINING PHASE ---- ####

for _ in training_bar:
    optimizer.zero_grad()

    loss = torch.sum(torch.sqrt(integral_rule.integrate(loss_function, f_value))) ** 2

    loss_value_float = loss.item()

    if loss_value_float < best_loss - MIN_DELTA:
        best_loss = loss_value_float
        EARLY_STOPPING_COUNTER = 0
        optimal_parameters = NN.state_dict()
    else:
        EARLY_STOPPING_COUNTER += 1
        if EARLY_STOPPING_COUNTER >= EARLY_STOPPING_PATIENCE:
            break

    loss.backward()

    optimizer.step()

    training_bar.set_description(f"Loss: {loss_value_float:.8e}")

    training_history.append(loss_value_float)

### ---- PLOTTING ---- ####

plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)
plot_points_np = plot_points.numpy(force=True)


fig_solution, ax_solution = plt.subplots()

ax_solution.plot(
    plot_points_np, f(plot_points).numpy(force=True), label="Exact function"
)
ax_solution.plot(
    plot_points_np,
    NN(plot_points).numpy(force=True),
    linestyle=":",
    label="NN approximation",
)
ax_solution.set_title("Approximation of the boundary condition")
ax_solution.set_xlabel("x")
ax_solution.set_ylabel("f(x)")
ax_solution.set_xlim(-0.01, 1.01)
ax_solution.set_ylim(-0.01, 1.01)
ax_solution.legend()

fig_loss, ax_loss = plt.subplots()

ax_loss.semilogy(training_history)
ax_loss.set_title("Training History")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss Value")


plt.show()
