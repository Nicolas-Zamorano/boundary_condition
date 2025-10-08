"""test C^2 boundary_layer"""

from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt


plot_points = torch.linspace(0, 1, 1000)


def function(x: torch.Tensor) -> torch.Tensor:
    """function"""
    value = torch.zeros_like(x)
    sign = torch.sign(x) > 0

    value[sign] = torch.exp(-1 / x[sign])

    return value


def s(x: torch.Tensor) -> torch.Tensor:
    """C^2 activation function"""
    value = function(x)

    return value / (value + function(1 - x))


def g_a(x: torch.Tensor, a: float) -> torch.Tensor:
    """C^2 activation function with desired boundary decay determined by a"""
    zeros = torch.zeros_like(x)
    sign = torch.sign((a / 3) - x)
    max_value_0 = torch.maximum(zeros, sign)
    max_value_1 = torch.maximum(zeros, -sign)

    value = 1.5 * a * max_value_0 + s((x + (a / 3)) / ((4 / 3) * a)) * max_value_1

    return value


def g(x: torch.Tensor, epsilons: Tuple[float, float, float]) -> torch.Tensor:
    """boundary constrain"""
    epsilon_0, epsilon_1, _ = epsilons

    return g_a(x, epsilon_0) + g_a(1 - x, epsilon_0 + epsilon_1) - 1


fig_1, ax_1 = plt.subplots()

ax_1.plot(plot_points, function(plot_points), label="f(x)")
ax_1.plot(plot_points, s(plot_points), label="S(x)")
ax_1.legend()

a_values = [0.25, 0.5, 0.75]

fig_2, ax_2 = plt.subplots()

for a_value in a_values:
    g_value = g_a(plot_points, a_value)
    ax_2.plot(plot_points, g_value, label=f"a={a_value}")

ax_2.legend()

e_0 = [1 / 3, 1 / 6, 5 / 7, 1 / 8]
e_1 = [1 / 3, 1 / 6, 1 / 7, 6 / 8]
e_2 = [1 / 3, 4 / 6, 1 / 7, 1 / 8]

fig, ax = plt.subplots()

line_styles = ["-", "--", "-.", ":"]

scaling = [1, 2, 3, 4]

for e in zip(e_0, e_1, e_2):

    values = scaling.pop(0) * g(plot_points, e)

    ax.plot(plot_points, values, label=f"{np.round(e,2)}", linestyle=line_styles.pop(0))

ax.legend()
plt.show()
