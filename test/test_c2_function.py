"""test C^2 boundary_layer"""

from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt


plot_points = torch.linspace(0, 1, 1000, requires_grad=True)


def function(x: torch.Tensor) -> torch.Tensor:
    """function"""
    return torch.where(x > 0, torch.exp(-1 / x), torch.tensor(0.0))


def s(x: torch.Tensor) -> torch.Tensor:
    """C^2 activation function"""
    return function(x) / (function(x) + function(1 - x))


def g_a(x: torch.Tensor, a: float) -> torch.Tensor:
    """C^2 activation function with desired boundary decay determined by a"""
    return torch.where(x < a / 3, 1.5 / a * x, s((x + (a / 3)) / ((4 / 3) * a)))


def g(x: torch.Tensor, epsilons: Tuple[float, float, float]) -> torch.Tensor:
    """boundary constrain"""
    return g_a(x, epsilons[0]) + g_a(1 - x, epsilons[2]) - 1


fig_1, ax_1 = plt.subplots()

function_value = function(plot_points)
function_jab = torch.autograd.grad(
    function_value, plot_points, torch.ones_like(function_value), create_graph=True
)[0]
function_hessian = torch.autograd.grad(
    function_jab, plot_points, torch.ones_like(function_value)
)[0]

s_value = s(plot_points)
s_jab = torch.autograd.grad(
    s_value, plot_points, torch.ones_like(s_value), create_graph=True
)[0]
s_hessian = torch.autograd.grad(s_jab, plot_points, torch.ones_like(s_value))[0]

ax_1.plot(plot_points.numpy(force=True), function_value.numpy(force=True), label="f(x)")
ax_1.plot(plot_points.numpy(force=True), function_jab.numpy(force=True), label="f'(x)")
ax_1.plot(
    plot_points.numpy(force=True), function_hessian.numpy(force=True), label="f''(x)"
)
ax_1.plot(plot_points.numpy(force=True), s_value.numpy(force=True), label="S(x)")
ax_1.plot(plot_points.numpy(force=True), s_jab.numpy(force=True), label="S'(x)")
ax_1.plot(plot_points.numpy(force=True), s_hessian.numpy(force=True), label="S''(x)")

ax_1.legend()

plt.show()

a_values = [0.33, 0.5, 0.75]

fig_2, ax_2 = plt.subplots()

for a_value in a_values:
    g_value = g_a(plot_points, a_value).numpy(force=True)
    ax_2.plot(plot_points.numpy(force=True), g_value, label=f"a={a_value}")

ax_2.legend()

e_0 = [1 / 3, 1 / 6, 5 / 7, 1 / 8]
e_1 = [1 / 3, 1 / 6, 1 / 7, 6 / 8]
e_2 = [1 / 3, 4 / 6, 1 / 7, 1 / 8]

fig, ax = plt.subplots()

line_styles = ["-", "--", "-.", ":"]


for e in zip(e_0, e_1, e_2):
    fig, ax = plt.subplots()

    values = scaling.pop(0) * g(plot_points, e)
    values = (g(plot_points, e)).numpy(force=True)

    ax.plot(
        plot_points.numpy(force=True),
        values,
        label=f"{np.round(e,2)}",
    )

    ax.legend()

plt.show()
