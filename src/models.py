"""Module for Neural Networks"""

from typing import List, Optional, Tuple
import torch


class C0BoundaryModel(torch.nn.Module):
    """NN to approximate the boundary condition."""

    def __init__(self):
        super(C0BoundaryModel, self).__init__()
        self.epsilons = torch.nn.Parameter(
            torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
        )

    def function_approx(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Approximation function."""
        weights_0, weights_1, weights_2 = torch.split(weights, 1, -1)
        return (
            torch.relu(x / weights_0)
            - torch.relu((x / weights_0) - 1)
            - torch.relu((x - weights_0 - weights_1) / weights_2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        weights = torch.softmax(self.epsilons, dim=-1)

        return self.function_approx(x, weights)


class C2BoundaryModel(torch.nn.Module):
    """NN to approximate boundary condition"""

    def __init__(self):
        super(C2BoundaryModel, self).__init__()
        self.epsilons = torch.nn.Parameter(
            torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
        )

    def function(self, x: torch.Tensor) -> torch.Tensor:
        """function"""
        value = torch.zeros_like(x)
        sign = torch.sign(x) > 0

        value[sign] = torch.exp(-1 / x[sign])

        return value

    def s(self, x: torch.Tensor) -> torch.Tensor:
        """C^2 activation function"""
        return self.function(x) / (self.function(x) + self.function(1 - x))

    def g_a(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """C^2 activation function with desired boundary decay determined by a"""
        sign = x < a / 3
        result = torch.zeros_like(x)
        result[sign] = 1.5 / a * x[sign]
        result[~sign] = self.s((x[~sign] + (a / 3)) / ((4 / 3) * a))
        return result

    def g(self, x: torch.Tensor, epsilons: torch.Tensor) -> torch.Tensor:
        """boundary constrain"""
        return self.g_a(x, epsilons[0]) + self.g_a(1 - x, epsilons[2]) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        weights = torch.softmax(self.epsilons, dim=-1)

        return self.g(x, weights)


class IdentityBC(torch.nn.Module):
    """base class for strong application of  boundary conditions"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass"""
        return torch.ones_like(x[..., :1])


class FeedForwardNeuralNetwork(torch.nn.Module):
    """Feed-Forward Neural Network (FNN) class compatible with TorchScript."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        nb_hidden_layers: int,
        neurons_per_layers: int,
        activation_function: torch.nn.Module = torch.nn.Tanh(),
        use_xavier_initialization: bool = False,
        boundary_condition_modifier: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._nb_hidden_layers = nb_hidden_layers
        self._neurons_per_layers = neurons_per_layers
        self._activation_function = activation_function
        self._use_xavier_initialization = use_xavier_initialization

        if boundary_condition_modifier is None:
            self._boundary_condition_modifier = IdentityBC()
        else:
            self._boundary_condition_modifier = boundary_condition_modifier

        self.neural_network = self.build_network(
            input_dimension,
            output_dimension,
            nb_hidden_layers,
            neurons_per_layers,
            activation_function,
            use_xavier_initialization,
        )

    def build_network(
        self,
        input_dimension: int,
        output_dimension: int,
        nb_layers: int,
        neurons_per_layers: int,
        activation_function: torch.nn.Module,
        use_xavier_initialization: bool,
    ) -> torch.nn.Sequential:
        """Build the neural network architecture."""
        layers = []

        layers.append(torch.nn.Linear(input_dimension, neurons_per_layers))
        layers.append(activation_function)

        for _ in range(nb_layers):
            layers.append(torch.nn.Linear(neurons_per_layers, neurons_per_layers))
            layers.append(activation_function)

        layers.append(torch.nn.Linear(neurons_per_layers, output_dimension))

        seq = torch.nn.Sequential(*layers)

        if use_xavier_initialization:
            for layer in seq:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)

        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.neural_network(x) * self._boundary_condition_modifier(x)

    def value_and_gradient(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the gradient of the neural network with respect to its inputs."""
        inputs.requires_grad_(True)
        output = self.forward(inputs)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]

        gradients = torch.autograd.grad(
            outputs=[output],
            inputs=[inputs],
            grad_outputs=grad_outputs,  # type: ignore
            retain_graph=True,
            create_graph=True,
        )[0]

        return output, gradients

    def value_and_laplacian(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the laplacian of the neural network with respect to its inputs."""
        inputs.requires_grad_(True)
        output = self.forward(inputs)

        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(output)]

        gradients: Optional[torch.Tensor] = torch.autograd.grad(
            outputs=[output],
            inputs=[inputs],
            grad_outputs=grad_outputs,  # type: ignore
            retain_graph=True,
            create_graph=True,
        )[0]

        assert gradients is not None

        laplacian = torch.zeros_like(output)

        for i in range(inputs.shape[-1]):

            gradient: torch.Tensor = gradients[..., i]
            gradient_outputs: List[Optional[torch.Tensor]] = [
                torch.ones_like(output).squeeze(-1)
            ]
            grad2 = torch.autograd.grad(
                [gradient],
                [inputs],
                grad_outputs=gradient_outputs,  # type: ignore
                create_graph=True,
                retain_graph=True,
            )[0]
            assert grad2 is not None
            laplacian += grad2[..., i : i + 1]

        return output, gradients, laplacian
