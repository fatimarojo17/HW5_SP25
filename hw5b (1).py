import numpy as np
import random as rnd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def colebrook(f, Re, rr):
    if f <= 0:  # Prevent invalid sqrt computation
        return np.inf  # Return a large value to avoid negative sqrt
    return 1 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))


def ff(Re, rr, CBEQN=False):
    if Re >= 4000 or CBEQN:
        initial_guess = 0.02  # Improved initial guess
        result = fsolve(colebrook, initial_guess, args=(Re, rr))
        if np.any(result <= 0):  # Ensure the friction factor is positive
            return np.nan
        return result[0]
    elif Re <= 2000:
        return 64 / Re
    else:
        CBff = ff(4000, rr, True)
        Lamff = ff(2000, rr, False)
        mean = Lamff + (CBff - Lamff) * (Re - 2000) / 2000
        sigma = 0.2 * mean
        return max(0, rnd.normalvariate(mean, sigma))  # Ensure f remains non-negative


def plotMoody(points=[]):
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

    for Re, f, trans in points:
        marker = '^' if trans else 'o'
        plt.scatter(Re, f, color='red', marker=marker, s=100)

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re = \frac{Vd}{\nu}$", fontsize=14)
    plt.ylabel(r"Friction factor $f = \frac{h_f}{(L/D) (V^2/2g)}$", fontsize=14)
    plt.text(2E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=14)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both', linestyle='-', linewidth=0.5)
    plt.show()

def plotMoody(points=[]):
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

    # **Ensure both types of points are plotted**
    for Re, f, trans in points:
        if trans:  # **Transition flow (Re between 2000 and 4000)**
            plt.scatter(Re, f, color='red', marker='^', s=100, label="Transition Flow")
        else:  # **Laminar or Turbulent flow**
            plt.scatter(Re, f, color='red', marker='o', s=100, label="Steady Flow")

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re = \frac{Vd}{\nu}$", fontsize=14)
    plt.ylabel(r"Friction factor $f = \frac{h_f}{(L/D) (V^2/2g)}$", fontsize=14)
    plt.text(2E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=14)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.show()


def main():
    points = []
    while True:
        try:
            d = float(input("Enter pipe diameter (in inches): "))
            epsilon = float(input("Enter pipe roughness (in micro-inches): ")) / 1E6
            Q = float(input("Enter flow rate (in gallons/min): "))

            nu = 1.004E-5  # Kinematic viscosity of water in ft^2/s
            V = (Q / (0.0408 * d ** 2))  # Velocity in ft/s
            Re = (V * d) / nu
            rr = epsilon / d
            f = ff(Re, rr)
            if np.isnan(f):  # Ensure valid results
                print("Error: Unable to compute friction factor.")
                continue

            trans = 2000 <= Re < 4000
            points.append((Re, f, trans))

            plotMoody(points)

            cont = input("Do you want to enter another set of parameters? (y/n): ").strip().lower()
            if cont != 'y':
                break
        except ValueError:
            print("Invalid input. Please enter numerical values.")


if __name__ == "__main__":
    main()
