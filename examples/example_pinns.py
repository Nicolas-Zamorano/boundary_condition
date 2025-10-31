"""Example of approximating the boundary condition of a function using a NN."""

from typing import Tuple
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ..src import C2BoundaryModel, FeedForwardNeuralNetwork as FNN, Integration


torch.autograd.set_detect_anomaly(True)
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

NB_HIDDEN_LAYERS = 3
NEURONS_PER_LAYERS = 10
LEARNING_RATE = 1e-3

NN_boundary_layer = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=NB_HIDDEN_LAYERS,
    neurons_per_layers=NEURONS_PER_LAYERS,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_layer,
)

NN_boundary_NN = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=NB_HIDDEN_LAYERS,
    neurons_per_layers=NEURONS_PER_LAYERS,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_NN,
)

optimizer_boundary_layer = torch.optim.Adam(
    NN_boundary_layer.parameters(), lr=LEARNING_RATE
)

optimizer_boundary_NN = torch.optim.Adam(
    (list(NN_boundary_NN.parameters()) + list(boundary_NN.parameters())),
    lr=LEARNING_RATE,
)

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


def rhs(x: torch.Tensor) -> torch.Tensor:
    """Right hand side"""

    return (
        SCALING_FACTOR * (torch.exp(-x / EPSILON) / EPSILON) * (((1 - x) / EPSILON) + 2)
    )


def loss_function(_, laplacian: torch.Tensor, value_rhs: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return (laplacian + value_rhs) ** 2


def h1_norm(_, value: torch.Tensor, value_dx: torch.Tensor) -> torch.Tensor:
    """compute h1 norm"""
    return (value) ** 2 + (value_dx) ** 2


integral_rule = Integration(
    intervals_start=0, interval_end=1, nb_intervals=100, integration_order=2
)

rhs_value = rhs(integral_rule.integration_points)
exact_value = exact(integral_rule.integration_points)
exact_dx_value = exact_dx(integral_rule.integration_points)
exact_norm = torch.sum(
    torch.sqrt(integral_rule.integrate(h1_norm, exact_value, exact_dx_value))
)

### ---- TRAINING PARAMETERS ---- ####


def training_step(
    neural_network: FNN, optimizer: torch.optim.Optimizer
) -> Tuple[float, float]:
    """Single training step."""
    optimizer.zero_grad()

    nn_evaluation, nn_gradient, nn_laplacian = neural_network.value_and_laplacian(
        integral_rule.integration_points
    )

    loss = torch.sum(integral_rule.integrate(loss_function, nn_laplacian, rhs_value))

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


EPOCHS = 10000

training_bar = tqdm(range(EPOCHS))

history_loss_boundary_layer = []
history_loss_boundary_NN = []
history_h1_norm_boundary_layer = []
history_h1_norm_boundary_NN = []
boundary_weights_history = []
grads_boundary_weights_history = []

### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    loss_boundary_layer, h1_norm_boundary_layer = training_step(
        NN_boundary_layer, optimizer_boundary_layer
    )

    loss_boundary_NN, h1_norm_boundary_NN = training_step(
        NN_boundary_NN, optimizer_boundary_NN
    )

    boundary_weights_history.append(boundary_NN.epsilons.data)
    grads_boundary_weights_history.append(boundary_NN.epsilons.grad)
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

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

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

axes[0, 3].semilogy(history_loss_boundary_NN, label=r"$\mathcal{L}(u_\theta)$")

axes[0, 3].semilogy(
    history_h1_norm_boundary_layer,
    label=r"$\frac{\|u - u_\theta\|_{H^1}}{\|u\|_{H^1}}$",
)
axes[0, 3].legend()
axes[0, 3].set_xlabel("# Epochs")
axes[0, 3].set_ylabel("Values")
axes[0, 3].set_title("Training History")

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

axes[1, 3].semilogy(history_loss_boundary_NN, label=r"$\mathcal{L}(u_\theta)$")
axes[1, 3].semilogy(
    history_h1_norm_boundary_NN,
    label=r"$\frac{\|u - u_\theta\|_{H^1}}{\|u\|_{H^1}}$",
)

fig.text(
    0.5, 0.9, "Non-Trainable Boundary Layer", ha="center", va="center", fontsize=14
)

fig.text(0.5, 0.48, "Trainable Boundary Layer", ha="center", va="center", fontsize=14)

#### --- Boundary Layer VALUES PLOT --- ###

weights_array = torch.stack(boundary_weights_history, dim=0).numpy()
grads_array = torch.stack(grads_boundary_weights_history, dim=0).numpy()

iterations = range(weights_array.shape[0])

line_styles = ["--", "-.", ":"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i in range(weights_array.shape[1]):
    axes[0].plot(
        iterations,
        weights_array[:, i],
        label=rf"$w_{i}$",
        linestyle=line_styles[i % len(line_styles)],
    )
    axes[1].plot(
        iterations,
        grads_array[:, i],
        label=rf"$\nabla_\theta w_{i}$",
        linestyle=line_styles[i % len(line_styles)],
    )

axes[0].set_title("Boundary NN weights evolution")
axes[0].set_xlabel("# Epochs")
axes[0].set_ylabel("Weight value")
axes[0].legend()
axes[1].set_title("Boundary NN weights gradients evolution")
axes[1].set_xlabel("# Epochs")
axes[1].set_ylabel("Gradient value")
axes[1].legend()
fig.suptitle("Boundary NN parameters evolution")
plt.show()
