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


boundary_constrain = C2BoundaryModel()

# boundary_constrain = BoundaryLayer()

### ---- NEURAL_NETWORK ---- ###

NN = FNN(
    input_dimension=1,
    output_dimension=1,
    nb_hidden_layers=3,
    neurons_per_layers=10,
    activation_function=torch.nn.Tanh(),
    use_xavier_initialization=True,
    boundary_condition_modifier=boundary_constrain,
)

optimizer = torch.optim.Adam(NN.parameters(), lr=1e-3)


### ---- LOSS PARAMETERS ---- ####

EPSILON = 1e-2
SCALING_FACTOR = 1


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
    intervals_start=0, interval_end=1, nb_intervals=70, integration_order=4
)

integral_rule_test = Integration(
    intervals_start=0, interval_end=1, nb_intervals=200, integration_order=8
)

integral_rule_validation = Integration(
    intervals_start=0, interval_end=1, nb_intervals=30, integration_order=4
)

rhs_value = rhs(integral_rule.integration_points)
exact_value = exact(integral_rule_validation.integration_points)
exact_dx_value = exact_dx(integral_rule_validation.integration_points)
exact_norm = torch.sum(
    torch.sqrt(integral_rule_validation.integrate(h1_norm, exact_value, exact_dx_value))
)

### ---- TRAINING PARAMETERS ---- ####

EPOCHS = 10000

training_bar = tqdm(range(EPOCHS))

history_loss = []
history_h1_norm = []

best_loss = float("inf")
MIN_DELTA = 1e-16
EARLY_STOPPING_PATIENCE = 200
EARLY_STOPPING_COUNTER = 0
optimal_parameters = NN.state_dict()

### ---- TRAINING PHASE ---- ####

for _ in training_bar:

    optimizer.zero_grad()

    nn_evaluation, nn_gradient, nn_laplacian = NN.value_and_laplacian(
        integral_rule.integration_points
    )

    loss = torch.sum(integral_rule.integrate(loss_function, nn_laplacian, rhs_value))

    nn_evaluation_validation, nn_gradient_validation = NN.value_and_gradient(
        integral_rule_validation.integration_points
    )

    error_h1 = (
        torch.sqrt(
            torch.sum(
                integral_rule_validation.integrate(
                    h1_norm,
                    exact_value - nn_evaluation_validation,
                    exact_dx_value - nn_gradient_validation,
                )
            )
        )
        / exact_norm
    )

    loss.backward()
    optimizer.step()

    loss_value_float = loss.item()
    error_h1_float = error_h1.item()

    history_loss.append(loss_value_float)
    history_h1_norm.append(error_h1_float)

    training_bar.set_postfix(
        {
            "Loss": f"{loss_value_float:.8e}",
            "H1 Error": f"{error_h1_float:.8e}",
        }
    )

    if loss_value_float < best_loss - MIN_DELTA:
        best_loss = loss_value_float
        EARLY_STOPPING_COUNTER = 0
        optimal_parameters = NN.state_dict()
    else:
        EARLY_STOPPING_COUNTER += 1
        if EARLY_STOPPING_COUNTER >= EARLY_STOPPING_PATIENCE:
            break

### ---- PLOTTING ---- ####

NN.load_state_dict(optimal_parameters)

exact_value_test = exact(integral_rule_test.integration_points)
exact_dx_value_test = exact_dx(integral_rule_test.integration_points)

nn_evaluation_test, nn_gradient_test = NN.value_and_gradient(
    integral_rule_test.integration_points
)

h1_norm_test = torch.sqrt(
    torch.sum(
        integral_rule_test.integrate(
            h1_norm,
            exact_value_test - nn_evaluation_test,
            exact_dx_value_test - nn_gradient_test,
        )
    )
) / torch.sqrt(
    torch.sum(
        integral_rule_test.integrate(h1_norm, exact_value_test, exact_dx_value_test)
    )
)

print(f"H1 Norm on test points: {h1_norm_test.item():.4e}")

plot_points = integral_rule_test.integration_points.reshape(-1, 1)
plot_points_np = plot_points.numpy(force=True)

solution_evaluation = exact(plot_points).numpy(force=True)

### --- BOUNDARY LAYER PLOT --- ###

boundary_layer_evaluation = boundary_constrain(plot_points).numpy(force=True)
NN_boundary_layer_evaluation = NN.neural_network(plot_points).numpy(force=True)

fig_boundary, ax_boundary = plt.subplots()

ax_boundary.plot(plot_points_np, solution_evaluation, label="Exact solution")
ax_boundary.plot(
    plot_points_np,
    boundary_layer_evaluation,
    label="Boundary Constraint",
    linestyle="--",
)

ax_boundary.legend()

fig_only_nn, ax_only_nn = plt.subplots()

ax_only_nn.plot(plot_points_np, solution_evaluation, label="Exact solution")

ax_only_nn.plot(
    plot_points_np,
    NN_boundary_layer_evaluation,
    label="Neural Network",
    linestyle="--",
)
ax_only_nn.legend()

fig_sol, ax_sol = plt.subplots()

ax_sol.plot(plot_points_np, solution_evaluation, label="Exact solution")

ax_sol.plot(
    plot_points_np,
    NN_boundary_layer_evaluation * boundary_layer_evaluation,
    label="NN * Boundary Constraint",
    linestyle="--",
)
ax_sol.legend()

fig_loss, ax_loss = plt.subplots()

ax_loss.semilogy(history_loss, label=r"$\mathcal{L}_h(u_\theta)$")

ax_loss.semilogy(
    history_h1_norm,
    label=r"$\frac{\|u - u_\theta\|_{H^1}}{\|u\|_{H^1}}$",
)
ax_loss.legend()
ax_loss.set_xlabel("# Epochs")
ax_loss.set_ylabel("Values")

plt.show()
