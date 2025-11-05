"""Example of approximating the boundary condition of a function using a NN."""

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from integration import Integration
from models import C2BoundaryModel, FeedForwardNeuralNetwork as FNN

torch.set_default_dtype(torch.float64)

### ---- BOUNDARY CONDITION ---- ###


class BoundaryLayer(torch.nn.Module):
    """Class to strongly apply bc"""

    def forward(self, inputs):
        """Boundary condition modifier function."""
        return inputs * (1 - inputs)


boundary_layer = BoundaryLayer()

NN = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_layer,
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


def loss_function(_, value: torch.Tensor, value_exact: torch.Tensor) -> torch.Tensor:
    """Loss function to minimize."""
    return (value - value_exact) ** 2


def h1_norm(_, value: torch.Tensor, value_dx: torch.Tensor) -> torch.Tensor:
    """compute h1 norm"""
    return (value) ** 2 + (value_dx) ** 2


integral_rule = Integration(
    intervals_start=0, interval_end=1, nb_intervals=100, integration_order=2
)

integral_rule_test = Integration(
    intervals_start=0, interval_end=1, nb_intervals=100, integration_order=4
)

integral_rule_validation = Integration(
    intervals_start=0, interval_end=1, nb_intervals=200, integration_order=6
)

exact_value = exact(integral_rule.integration_points)
exact_value_validation = exact(integral_rule_validation.integration_points)
exact_dx_value = exact_dx(integral_rule.integration_points)
exact_norm = torch.sqrt(
    torch.sum(integral_rule.integrate(h1_norm, exact_value, exact_dx_value))
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 10000

training_bar = tqdm(range(EPOCHS))

history_loss_boundary_layer = []
history_h1_norm_boundary_layer = []


def training_step(
    neural_network: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    """Training step."""
    optimizer.zero_grad()

    nn_evaluation, nn_gradient = neural_network.value_and_gradient(
        integral_rule.integration_points
    )

    nn_evaluation_validation = neural_network.value_and_gradient(
        integral_rule_validation.integration_points)

    loss = torch.sum(integral_rule.integrate(loss_function, nn_evaluation, exact_value))

    loss_validation = torch.sum(integral_rule_validation.integrate(loss_function, nn_evaluation_validation)

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

### ---- NEURAL_NETWORK ---- ###

optimizer_boundary_layer = torch.optim.Adam(NN_boundary_layer.parameters(), lr=1e-3)


### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    loss_boundary_layer, h1_norm_boundary_layer = training_step(
        NN_boundary_layer, optimizer_boundary_layer
    )

    history_loss_boundary_layer.append(loss_boundary_layer)
    history_h1_norm_boundary_layer.append(h1_norm_boundary_layer)

    training_bar.set_postfix(
        {
            "Loss": f"{loss_boundary_layer:.4e}",
            "H1 Error": f"{h1_norm_boundary_layer:.4e}",
        }
    )

### ---- PLOTTING ---- ####

plot_points = torch.linspace(0, 1, 1000).unsqueeze(1)
plot_points_np = plot_points.numpy(force=True)

solution_evaluation = exact(plot_points).numpy(force=True)

### --- BOUNDARY LAYER PLOT --- ###

boundary_layer_evaluation = boundary_layer(plot_points).numpy(force=True)
NN_boundary_layer_evaluation = NN_boundary_layer.neural_network(plot_points).numpy(
    force=True
)

# Plot 1: Boundary Layer Function B(x)
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(plot_points_np, solution_evaluation, label=r"$u_\text{ex}(x)$")
ax1.plot(
    plot_points_np,
    boundary_layer_evaluation,
    label=r"$B(x)$",
    linestyle="--",
)
ax1.legend()
ax1.set_xlabel("x")
ax1.set_ylabel("Value")
ax1.set_title("Boundary Layer Function")
ax1.grid(True, alpha=0.3)

# Plot 2: Neural Network Output
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(plot_points_np, solution_evaluation, label=r"$u_\text{ex}(x)$")
ax2.plot(
    plot_points_np,
    NN_boundary_layer_evaluation,
    label=r"$u_{\theta}(x)$",
    linestyle="--",
)
ax2.legend()
ax2.set_xlabel("x")
ax2.set_ylabel("Value")
ax2.set_title("Neural Network Output")
ax2.grid(True, alpha=0.3)

# Plot 3: Final Approximation
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(plot_points_np, solution_evaluation, label=r"$u_\text{ex}(x)$")
ax3.plot(
    plot_points_np,
    NN_boundary_layer_evaluation * boundary_layer_evaluation,
    label=r"$B(x)\,u_{\theta}(x)$",
    linestyle="--",
)
ax3.legend()
ax3.set_xlabel("x")
ax3.set_ylabel("Value")
ax3.set_title("Final Approximation with Boundary Condition")
ax3.grid(True, alpha=0.3)

# Plot 4: Training Loss
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.semilogy(history_loss_boundary_layer, label=r"$\mathcal{L}(Bu_\theta)$")
ax4.legend()
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.set_title("Training Loss")
ax4.grid(True, alpha=0.3)

# Plot 5: H1 Relative Error
fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.semilogy(
    history_h1_norm_boundary_layer,
    label=r"$\frac{\|u - Bu_\theta\|_{H^1}}{\|u\|_{H^1}}$",
)
ax5.legend()
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Relative H1 Error")
ax5.set_title("H1 Relative Error")
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
