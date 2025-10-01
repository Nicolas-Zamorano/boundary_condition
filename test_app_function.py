"""test to see different weights for the boundary approximation function"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from models import BoundaryModel

model = BoundaryModel(nb_points=3)


e_0 = [1 / 3, 1 / 6, 5 / 7, 1 / 8]
e_1 = [1 / 3, 1 / 6, 1 / 7, 6 / 8]
e_2 = [1 / 3, 4 / 6, 1 / 7, 1 / 8]

points = torch.linspace(0, 1, 1000)

fig, ax = plt.subplots()

line_styles = ["-", "--", "-.", ":"]

scaling = [1, 2, 3, 4]

for e in zip(e_0, e_1, e_2):

    model.layer.weight.data = torch.tensor([e])

    values = scaling.pop(0) * model(points).detach().squeeze().numpy()

    ax.plot(points, values, label=f"{np.round(e,2)}", linestyle=line_styles.pop(0))

ax.legend()
plt.show()
