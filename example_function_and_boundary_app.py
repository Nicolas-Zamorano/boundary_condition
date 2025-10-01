"""Example of approximating the boundary condition of a function using a NN."""

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import BoundaryModel, FeedForwardNeuralNetwork as FNN
from integration import Integration

### ---- BOUNDARY CONDITION ---- ###


class BoundaryLayer(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        return inputs * (1 - inputs)


boundary_NN = BoundaryModel()

boundary_layer = BoundaryLayer()

### ---- NEURAL_NETWORK ---- ###

NN_boundary_layer = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
)

NN_boundary_NN = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
)

optimizer_boundary_layer = torch.optim.Adam(NN_boundary_layer.parameters(), lr=1e-3)

optimizer_boundary_NN = torch.optim.Adam(
    list(NN_boundary_NN.parameters()) + list(boundary_NN.parameters()), lr=1e-3
)

### ---- LOSS PARAMETERS ---- ####

EPSILON = 1e-2


def f(x: torch.Tensor) -> torch.Tensor:
    """Function to approximate his boundary condition."""
    y = (1 - torch.exp(-x / EPSILON)) * (1 - x)
    max_val, _ = torch.max(y, dim=0)
    return y / max_val


def loss_function(
    points: torch.Tensor,
    neural_network: torch.nn.Module,
    boundary_function: torch.nn.Module,
) -> torch.Tensor:
    """Loss function to minimize."""
    return (neural_network(points) * boundary_function(points) - f(points)) ** 2


integral_rule = Integration(
    intervals_start=0, interval_end=1, nb_intervals=100, integration_order=2
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 6000

training_bar = tqdm(range(EPOCHS))


history_loss_boundary_layer = []
history_loss_boundary_NN = []

### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    ### --- TRAINING BOUNDARY LAYER --- ###

    optimizer_boundary_layer.zero_grad()
    loss_boundary_layer = (
        torch.sum(
            torch.sqrt(
                integral_rule.integrate(
                    loss_function, NN_boundary_layer, boundary_layer
                )
            )
        )
        ** 2
    )
    loss_boundary_layer.backward()
    optimizer_boundary_layer.step()
    loss_boundary_layer_float = loss_boundary_layer.item()
    history_loss_boundary_layer.append(loss_boundary_layer_float)

    ### --- TRAINING BOUNDARY LAYER --- ###

    optimizer_boundary_NN.zero_grad()
    loss_boundary_NN = (
        torch.sum(
            torch.sqrt(
                integral_rule.integrate(loss_function, NN_boundary_NN, boundary_NN)
            )
        )
        ** 2
    )
    loss_boundary_NN.backward()
    optimizer_boundary_NN.step()
    loss_boundary_NN_float = loss_boundary_NN.item()
    history_loss_boundary_NN.append(loss_boundary_NN_float)

    training_bar.set_description(
        f"Strong Loss: {loss_boundary_layer_float:.4e} Weak Loss: {loss_boundary_NN_float:.4e}"
    )


### ---- PLOTTING ---- ####

plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)
plot_points_np = plot_points.numpy(force=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

solution_evaluation = f(plot_points).numpy(force=True)

### --- BOUNDARY LAYER PLOT --- ###

boundary_layer_evaluation = boundary_layer(plot_points).numpy(force=True)
NN_boundary_layer_evaluation = NN_boundary_layer(plot_points).numpy(force=True)

axes[0, 0].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[0, 0].plot(
    plot_points_np,
    boundary_layer_evaluation,
    label="Boundary Constraint",
    linestyle="--",
)

axes[0, 0].legend()

axes[0, 1].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[0, 1].plot(
    plot_points_np,
    NN_boundary_layer_evaluation,
    label="Neural Network",
    linestyle="--",
)
axes[0, 1].legend()

axes[0, 2].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[0, 2].plot(
    plot_points_np,
    NN_boundary_layer_evaluation * boundary_layer_evaluation,
    label="NN * Boundary Constraint",
    linestyle="--",
)
axes[0, 2].legend()

### --- BOUNDARY NN PLOT --- ###

boundary_NN_evaluation = boundary_NN(plot_points).numpy(force=True)
NN_boundary_NN_evaluation = NN_boundary_NN(plot_points).numpy(force=True)

axes[1, 0].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[1, 0].plot(
    plot_points_np,
    boundary_NN_evaluation,
    label="Boundary Constraint",
    linestyle="--",
)
axes[1, 0].legend()

axes[1, 1].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[1, 1].plot(
    plot_points_np, NN_boundary_NN_evaluation, label="Neural Network", linestyle="--"
)
axes[1, 1].legend()

axes[1, 2].plot(plot_points_np, solution_evaluation, label="Exact solution")

axes[1, 2].plot(
    plot_points_np,
    NN_boundary_NN_evaluation * boundary_NN_evaluation,
    label="NN * Boundary Constraint",
    linestyle="--",
)
axes[1, 2].legend()

fig.text(
    0.5, 0.9, "Non-Trainable Boundary Layer", ha="center", va="center", fontsize=14
)

fig.text(0.5, 0.48, "Trainable Boundary Layer", ha="center", va="center", fontsize=14)

# Loss plot
# fig_loss, ax_loss = plt.subplots()
# ax_loss.semilogy(history_loss_boundary_NN, label="Strong Loss")
# ax_loss.semilogy(history_loss_boundary_layer, label="Weak Loss")
# ax_loss.legend()
# ax_loss.set_title("Training Losses")

plt.show()
