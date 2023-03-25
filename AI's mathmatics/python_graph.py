import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1001)
fx = x*x - 6 * np.sqrt(x+1)
plt.axis([-10, 10, -10, 10])
plt.plot(x, fx)
plt.show()