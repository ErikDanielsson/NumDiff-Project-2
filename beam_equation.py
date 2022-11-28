import matplotlib.pyplot as plt
import numpy as np

from twopBVP import twopBVP

# M''
# q(x) = -50000
# I(x) = 10e-3 * (3 - 2 * np.cos(np.pi*x/L) ** 12)
# '''

q = -50
L = 10
I = lambda x: 10e-3 * (3 - 2 * np.cos(np.pi * x / L) ** 12)
E = 1.9e8  # strong steel y'know
N = 999
grid = np.linspace(0, L, N + 2)
qvec = np.array([q for _ in grid[1:-1]])
M = twopBVP(qvec, 0, 0, L, N)
u = twopBVP(
    np.array([M[i + 1] / (E * I(x)) for i, x in enumerate(grid[1:-1])]), 0, 0, L, N
)
plt.plot(grid, u)
plt.xlabel("Distance (m)")
plt.ylabel("Deflection from midline (m)")
plt.show()
