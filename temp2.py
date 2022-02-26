import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal

data = multivariate_normal([10, 10], [[3, 2], [2, 3]], size=10000)

plt.hist2d(data[:, 0], data[:, 1], bins=100)
plt.show()
