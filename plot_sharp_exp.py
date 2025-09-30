"""Plot test function"""

import numpy as np
import matplotlib.pyplot as plt

points = np.linspace(0, 1, 1000)

EPSILON = 1e-1

function_value = (1 - np.exp(-points / EPSILON)) * (1 - points)

plt.plot(points, function_value)

plt.show()
