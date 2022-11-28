import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def laplace_1d(L, N):
    dx = L / N
    c = np.zeros(N)
    c[0] = -2
    c[1] = 1
    T = scipy.linalg.toeplitz(c, c)
    T[-1, -2] = 2
    T /= dx * dx
    e, S = np.linalg.eig(T)
    sorted_eigs = (
        (e, s) for e, s in sorted(zip(e, [S[:, i] for i in range(N)]), reverse=True)
    )
    e, S = map(list, zip(*sorted_eigs))
    t = np.linspace(0, L, N)
    plt.plot(t, S[0] / S[0][-1], label="$\phi_1$")
    plt.plot(t, -S[1] / S[0][-1], label="$\phi_2$")
    plt.plot(t, -S[2] / S[0][-1], label="$\phi_3$")
    plt.title("Eigenmodes for $\\frac{d^2}{dx^2}$ with $y(0) = y'(1) = 0$")
    plt.legend()

    plt.show()


laplace_1d(1, 500)
