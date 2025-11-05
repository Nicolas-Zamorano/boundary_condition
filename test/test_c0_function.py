"""test to see different weights for the boundary approximation function"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models import C0BoundaryModel

model = C0BoundaryModel()


e_0 = [1 / 3, 1 / 6, 5 / 7, 1 / 8]
e_1 = [1 / 3, 1 / 6, 1 / 7, 6 / 8]
e_2 = [1 / 3, 4 / 6, 1 / 7, 1 / 8]

points = torch.linspace(0, 1, 1000)


scaling = [1, 2, 3, 4]

for e in zip(e_0, e_1, e_2):

    fig, ax = plt.subplots()

    model.epsilons.data = torch.tensor([e])

    values = model(points).detach().squeeze().numpy()

    ax.plot(
        points,
        values,
        label=f"{np.round(e,2)}",
    )
    ax.set_xlabel("x")
    ax.set_ylabel(r"$B_\omega(x)$")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-0.01, 1.01)
    ax.legend()
    plt.show()
