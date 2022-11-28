import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def twopBVP(fvec, alpha, beta, L, N):
    dx = L / (N + 1)
    b = dx * dx * fvec
    b[0] -= alpha
    b[-1] -= beta
    c = np.zeros(N)
    c[0] = -2
    c[1] = 1
    y = scipy.linalg.solve_toeplitz((c, c), b)
    return np.concatenate(([alpha], y, [beta]))


def twopBVP_Neumann(fvec, alpha, beta, L, N):
    dx = L / N
    c = np.zeros(N)
    c[0] = -2
    c[1] = 1
    T = scipy.linalg.toeplitz(c, c)
    T[-1, -2] = 2
    print(T)


"""
f = lambda x: np.sin(x)
# y''=x^3 => y = x^5/20+cx+d. This is
N = 40
L = 4 * np.pi
alpha = 0
beta = np.pi
grid = np.linspace(0, L, N + 2)
fvec = np.array([f(x) for x in grid[1:-1]])

# y = twopBVP(fvec, alpha, beta, L, N)

# plt.scatter(grid, y, label="Finite difference method", c="r", marker="x")
# xs = np.linspace(0, L, 1000)
# plt.plot(xs, [-np.sin(x) + x / 4 for x in xs], label="Analytical solution")
# plt.legend()

# plt.show()

N_min = 10
N_max = 1000
Ns = list(range(N_min, N_max, 10))
global_errors = []
for N in Ns:
    grid = np.linspace(0, L, N + 2)
    fvec = np.array([f(x) for x in grid[1:-1]])
    y = twopBVP(fvec, alpha, beta, L, N)
    sol = [-np.sin(x) + x / 4 for x in grid]
    error = y - sol
    dx = L / (N + 1)
    global_errors.append(np.linalg.norm(error) * np.sqrt(dx))

plt.title("Convergence of finite difference solver")
plt.plot([L / (N + 1) for N in Ns], global_errors)
plt.yscale("log")
plt.ylabel("$||e||_{RMS}$")
plt.xscale("log")
plt.xlabel("$\Delta x$")
plt.show()
"""
