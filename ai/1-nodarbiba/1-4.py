import matplotlib.pyplot as plt

import numpy as np


def f(x):
    return np.sin(2 * x) + 2 * np.e ** (3 * x)


x_arr = np.linspace(start=-2, stop=2, num=1000)
y_arr = f(x_arr)

plt.plot(x_arr, y_arr)
plt.grid(True)
plt.show()
