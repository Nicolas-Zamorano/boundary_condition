"""Module for numerical integration"""

from typing import Tuple, Callable, Any
import torch
from numpy.polynomial.legendre import leggauss


class Integration:
    """Class to compute the integral of a function using composite Legendre-Gauss
    quadrature rule."""

    def __init__(
        self,
        intervals_start: float,
        interval_end: float,
        nb_intervals: int,
        integration_order: int = 2,
    ):
        self._intervals_start = intervals_start
        self.intervals_ends = interval_end
        self._nb_points = nb_intervals
        self._integration_order = integration_order
        self.intervals_points = torch.linspace(
            intervals_start, interval_end, nb_intervals + 1
        ).reshape(-1, 1, 1)

        self._integration_points, self._weights = self._compute_integral_values(
            self.intervals_points, self._integration_order
        )

    def _compute_integral_values(
        self, intervals_points: torch.Tensor, integral_order: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps the points and  weights to the intervals."""
        reference_integration_points, reference_weights = leggauss(integral_order)

        reference_integration_points = torch.tensor(
            reference_integration_points, dtype=torch.float32
        ).reshape(1, -1, 1)
        reference_weights = torch.tensor(
            reference_weights, dtype=torch.float32
        ).reshape(1, -1, 1)

        intervals_length = intervals_points[1:] - intervals_points[:-1]
        intervals_points_sum = intervals_points[1:] + intervals_points[:-1]

        mapped_integration_points = (
            0.5 * intervals_length * reference_integration_points
            + 0.5 * intervals_points_sum
        )
        mapped_weights = 0.5 * intervals_length * reference_weights

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
