import numpy as np

points = np.linspace(0, 1, 100)

value = points * (1 - points)

import matplotlib.pyplot as plt

plt.plot(points, value)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Boundary Condition Modifier Function")
plt.grid()
plt.show()
