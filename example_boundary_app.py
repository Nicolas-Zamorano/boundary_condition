"""Example of approximating the boundary condition of a function using a NN."""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import BoundaryModel
from integration import Integration

EPSILON = 2e-2


def f(x: torch.Tensor) -> torch.Tensor:
    """Function to approximate his boundary condition."""
    return (1 - torch.exp(-x / EPSILON)) * (1 - x)


class BoundaryConstrain(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        return inputs * (inputs - 1)


NN = BoundaryModel(nb_points=3)

integral_rule = Integration(a=0, b=1, nb_points=15, integration_order=2)

EPOCHS = 100

optimizer = torch.optim.Adam(NN.parameters(), lr=1e-1)


def loss_function(points: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return (NN(points) - f(points)) ** 2


training_bar = tqdm(range(EPOCHS))

training_history = []

best_loss = float("inf")
MIN_DELTA = 1e-16
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_COUNTER = 0

for _ in training_bar:
    optimizer.zero_grad()

    loss = torch.sum(torch.sqrt(integral_rule.integrate(loss_function))) ** 2

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

    training_bar.set_description(f"Loss: {loss_value_float:.4e}")

    training_history.append(loss_value_float)


plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)

fig_solution, ax_solution = plt.subplots()

ax_solution.plot(
    plot_points.numpy(), f(plot_points).detach().numpy(), label="Exact function"
)
ax_solution.plot(
    plot_points.numpy(), NN(plot_points).detach().numpy(), label="NN approximation"
)
fig_solution.legend()

fig_loss, ax_loss = plt.subplots()

ax_loss.semilogy(training_history)

plt.show()
