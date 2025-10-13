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

torch.set_default_dtype(torch.float64)

### ---- BOUNDARY CONDITION ---- ###


class BoundaryLayer(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        return inputs * (1 - inputs)


boundary_NN = C2BoundaryModel()

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
    return SCALING_FACTOR * (torch.exp(-x / EPSILON) * (1 + (1 - x / EPSILON) - 1))


def rhs(x: torch.Tensor) -> torch.Tensor:
    """Right hand side"""

    return SCALING_FACTOR * torch.exp(-x / EPSILON) * EPSILON * (2 + (1 - x) / EPSILON)


def loss_function(_, laplacian: torch.Tensor, value_rhs: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return laplacian + value_rhs


def h1_norm(_, value: torch.Tensor, value_dx: torch.Tensor) -> torch.Tensor:
    """compute h1 norm"""
    return (value) ** 2 + (value_dx) ** 2


def h1_error(
    _,
    value: torch.Tensor,
    grad: torch.Tensor,
    value_exact: torch.Tensor,
    value_dx_exact: torch.Tensor,
) -> torch.Tensor:
    """Compute h1 error"""
    return (value_exact - value) ** 2 + (value_dx_exact - grad) ** 2


integral_rule = Integration(
    intervals_start=0, interval_end=1, nb_intervals=15, integration_order=2
)

rhs_value = rhs(integral_rule.integration_points)
exact_value = exact(integral_rule.integration_points)
exact_dx_value = exact_dx(integral_rule.integration_points)
exact_norm = torch.sum(
    torch.sqrt(integral_rule.integrate(h1_norm, exact_value, exact_dx_value))
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 6000

training_bar = tqdm(range(EPOCHS))

history_loss_boundary_layer = []
history_loss_boundary_NN = []
history_h1_norm_boundary_layer = []
history_h1_norm_boundary_NN = []


### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    ### --- TRAINING BOUNDARY LAYER --- ###
    optimizer_boundary_layer.zero_grad()

    NN_boundary_layer_eval, NN_boundary_layer_grad, NN_boundary_layer_lap = (
        NN_boundary_layer.value_and_laplacian(integral_rule.integration_points)
    )

    loss_boundary_layer = (
        torch.sum(
            integral_rule.integrate(loss_function, NN_boundary_layer_lap, rhs_value)
        )
        ** 2
    )

    error_h1_boundary_layer = (
        torch.sum(
            torch.sqrt(
                integral_rule.integrate(
                    h1_error,
                    NN_boundary_layer_eval,
                    NN_boundary_layer_grad,
                    exact_value,
                    exact_dx_value,
                )
            )
        )
        / exact_norm
    )

    loss_boundary_layer.backward()
    optimizer_boundary_layer.step()

    loss_boundary_layer_float = loss_boundary_layer.item()
    h1_norm_boundary_layer_float = error_h1_boundary_layer.item()

    history_loss_boundary_layer.append(loss_boundary_layer_float)
    history_h1_norm_boundary_layer.append(h1_norm_boundary_layer_float)

    ### --- TRAINING BOUNDARY NN --- ###

    optimizer_boundary_NN.zero_grad()

    NN_boundary_NN_eval, NN_boundary_NN_grad, NN_boundary_NN_lap = (
        NN_boundary_NN.value_and_laplacian(integral_rule.integration_points)
    )

    loss_boundary_NN = (
        torch.sum(integral_rule.integrate(loss_function, NN_boundary_NN_lap, rhs_value))
        ** 2
    )

    error_h1_boundary_NN = (
        torch.sum(
            torch.sqrt(
                integral_rule.integrate(
                    h1_error,
                    NN_boundary_NN_eval,
                    NN_boundary_NN_grad,
                    exact_value,
                    exact_dx_value,
                )
            )
        )
        / exact_norm
    )

    loss_boundary_NN.backward()
    optimizer_boundary_NN.step()

    loss_boundary_NN_float = loss_boundary_NN.item()
    h1_norm_boundary_NN_float = error_h1_boundary_NN.item()

    history_loss_boundary_NN.append(loss_boundary_NN_float)
    history_h1_norm_boundary_NN.append(h1_norm_boundary_NN_float)

    training_bar.set_postfix(
        {
            "Loss Layer": f"{loss_boundary_layer_float:.4e}",
            "H1 Layer": f"{h1_norm_boundary_layer_float:.4e}",
            "Loss NN": f"{loss_boundary_NN_float:.4e}",
            "H1 NN": f"{h1_norm_boundary_NN_float:.4e}",
        }
    )


### ---- PLOTTING ---- ####

plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)
plot_points_np = plot_points.numpy(force=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

solution_evaluation = exact(plot_points).numpy(force=True)

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
fig_loss, ax_loss = plt.subplots()
ax_loss.semilogy(history_loss_boundary_NN, label="Strong Loss")
ax_loss.semilogy(history_loss_boundary_layer, label="Weak Loss")
ax_loss.semilogy(history_h1_norm_boundary_layer, label="Strong H1 error")
ax_loss.semilogy(history_h1_norm_boundary_NN, label="Weak H1 error")
ax_loss.legend()
ax_loss.set_title("Training Losses")

plt.show()
