"""Example of approximating the boundary condition of a function using a NN."""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from boundary_condition import (
    C0BoundaryModel,
    FeedForwardNeuralNetwork as FNN,
    Integration,
)

torch.set_default_dtype(torch.float64)

### ---- BOUNDARY CONDITION ---- ###


class BoundaryLayer(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        return inputs * (1 - inputs)


boundary_NN = C0BoundaryModel()

boundary_layer = BoundaryLayer()

### ---- NEURAL_NETWORK ---- ###

NN_boundary_layer = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_layer,
)

NN_boundary_NN = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_NN,
)

optimizer_boundary_layer = torch.optim.Adam(NN_boundary_layer.parameters(), lr=1e-3)

optimizer_boundary_NN = torch.optim.Adam(list(NN_boundary_NN.parameters()), lr=1e-3)

### ---- LOSS PARAMETERS ---- ####

EPSILON = 1e-2
SCALING_FACTOR = 1.1


def exact(x: torch.Tensor) -> torch.Tensor:
    """Function to approximate his boundary condition."""
    return SCALING_FACTOR * (1 - torch.exp(-x / EPSILON)) * (1 - x)


def exact_dx(x: torch.Tensor) -> torch.Tensor:
    """Derivate from exact"""
    exponential_value = torch.exp(-x / EPSILON)
    return SCALING_FACTOR * (
        (exponential_value / EPSILON) * (1 - x) - (1 - exponential_value)
    )


def loss_function(_, value: torch.Tensor, value_exact: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return (value - value_exact) ** 2


def h1_norm(_, value: torch.Tensor, value_dx: torch.Tensor) -> torch.Tensor:
    """compute h1 norm"""
    return (value) ** 2 + (value_dx) ** 2


integral_rule = Integration(
    intervals_start=0, interval_end=1, nb_intervals=100, integration_order=2
)

exact_value = exact(integral_rule.integration_points)
exact_dx_value = exact_dx(integral_rule.integration_points)
exact_norm = torch.sqrt(
    torch.sum(integral_rule.integrate(h1_norm, exact_value, exact_dx_value))
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 10000

training_bar = tqdm(range(EPOCHS))

history_loss_boundary_layer = []
history_loss_boundary_NN = []
history_h1_norm_boundary_layer = []
history_h1_norm_boundary_NN = []


def training_step(
    neural_network: FNN, optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """Training step."""
    optimizer.zero_grad()

    nn_evaluation, nn_gradient = neural_network.value_and_gradient(
        integral_rule.integration_points
    )

    loss = torch.sum(integral_rule.integrate(loss_function, nn_evaluation, exact_value))

    error_h1 = (
        torch.sqrt(
            torch.sum(
                integral_rule.integrate(
                    h1_norm,
                    exact_value - nn_evaluation,
                    exact_dx_value - nn_gradient,
                )
            )
        )
        / exact_norm
    )

    loss.backward()
    optimizer.step()

    return loss.item(), error_h1.item()


### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    loss_boundary_layer, h1_norm_boundary_layer = training_step(
        NN_boundary_layer, optimizer_boundary_layer
    )

    loss_boundary_NN, h1_norm_boundary_NN = training_step(
        NN_boundary_NN, optimizer_boundary_NN
    )

    history_loss_boundary_layer.append(loss_boundary_layer)
    history_h1_norm_boundary_layer.append(h1_norm_boundary_layer)
    history_loss_boundary_NN.append(loss_boundary_NN)
    history_h1_norm_boundary_NN.append(h1_norm_boundary_NN)

    training_bar.set_postfix(
        {
            "Loss Layer": f"{loss_boundary_layer:.4e}",
            "H1 Layer": f"{h1_norm_boundary_layer:.4e}",
            "Loss NN": f"{loss_boundary_NN:.4e}",
            "H1 NN": f"{h1_norm_boundary_NN:.4e}",
        }
    )

### ---- PLOTTING ---- ####

plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)
plot_points_np = plot_points.numpy(force=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

solution_evaluation = exact(plot_points).numpy(force=True)

### --- BOUNDARY LAYER PLOT --- ###

boundary_layer_evaluation = boundary_layer(plot_points).numpy(force=True)
NN_boundary_layer_evaluation = NN_boundary_layer.neural_network(plot_points).numpy(
    force=True
)

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
NN_boundary_NN_evaluation = NN_boundary_NN.neural_network(plot_points).numpy(force=True)

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
fig_loss, ax_loss = plt.subplots()
ax_loss.semilogy(history_loss_boundary_NN, label="Strong Loss")
ax_loss.semilogy(history_loss_boundary_layer, label="Weak Loss")
ax_loss.semilogy(history_h1_norm_boundary_layer, label="Strong H1 error")
ax_loss.semilogy(history_h1_norm_boundary_NN, label="Weak H1 error")
ax_loss.legend()
ax_loss.set_title("Training Losses")

plt.show()
