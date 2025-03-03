import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def colebrook(f, Re, rr):
    return 1 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))


def ff(Re, rr, CBEQN=False):
    if CBEQN:
        initial_guess = 0.02
        result = fsolve(colebrook, initial_guess, args=(Re, rr))
        return result[0]
    else:
        return 64 / Re


def plotMoody():
    ReValsCB = np.logspace(3.6, 8, 100)
    ReValsL = np.logspace(np.log10(600), np.log10(2000), 20)
    ReValsTrans = np.logspace(np.log10(2000), np.log10(4000), 20)

    rrVals = np.array([0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4,
                       1E-3, 2E-3, 4E-3, 6E-3, 8E-3, 1.5E-2, 2E-2, 3E-2, 4E-2, 5E-2])

    ffLam = np.array([ff(Re, 0, False) for Re in ReValsL])
    ffTrans = np.array([ff(Re, 0, False) for Re in ReValsTrans])
    ffCB = np.array([[ff(Re, relRough, True) for Re in ReValsCB] for relRough in rrVals])

    plt.figure(figsize=(10, 7))
    plt.loglog(ReValsL, ffLam, 'k', linewidth=1.5)
    plt.loglog(ReValsTrans, ffTrans, 'k--', linewidth=1.5)

    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], 'k', linewidth=1.2)
        plt.annotate(text=f"{rrVals[nRelR]}", xy=(ReValsCB[-1], ffCB[nRelR][-1]), fontsize=8)

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re = \frac{Vd}{\nu}$", fontsize=14)
    plt.ylabel(r"Friction factor $f = \frac{h_f}{(L/D) (V^2/2g)}$", fontsize=14)
    plt.text(2E8, 0.02, r"Relative roughness $rac{\epsilon}{d}$", rotation=90, fontsize=14)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both', linestyle='-', linewidth=0.5)
    plt.show()


def main():
    plotMoody()


if __name__ == "__main__":
    main()
