"""Module for numerical integration"""

from typing import Tuple, Callable, Any
import torch
from numpy.polynomial.legendre import leggauss


class Integration:
    """Class to compute the integral of a function using Legendre-Gauss."""

    def __init__(self, a: float, b: float, nb_points: int, integration_order: int = 2):
        self._a = a
        self._b = b
        self._nb_points = nb_points
        self._integration_order = integration_order
        self._points = torch.linspace(a, b, nb_points).reshape(-1, 1, 1)

        self._integration_points, self._weights = self._compute_integral_values(
            self._points, self._integration_order
        )

    def _compute_integral_values(
        self, points: torch.Tensor, integral_order: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the integral points and weights using Legendre-Gauss."""
        integration_points, weights = leggauss(integral_order)

        integration_points = torch.tensor(
            integration_points, dtype=torch.float32
        ).reshape(1, -1, 1)
        weights = torch.tensor(weights, dtype=torch.float32).reshape(1, -1, 1)

        difference_points = points[1:] - points[:-1]
        sum_points = points[1:] + points[:-1]

        mapped_integration_points = (
            0.5 * difference_points * integration_points + 0.5 * sum_points
        )
        mapped_weights = 0.5 * difference_points * weights

        return mapped_integration_points, mapped_weights

    def integrate(
        self,
        function: Callable[..., torch.Tensor],
        *args: Any | None,
        **kwargs: Any | None
    ) -> torch.Tensor:
        """Compute the integral of a function."""
        function_values = function(self._integration_points, *args, **kwargs)
        integral = torch.sum(function_values * self._weights, dim=-2)
        return integral
