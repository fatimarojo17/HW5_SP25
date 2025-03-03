# region imports 
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# endregion

# region functions
def ff(Re, rr, CBEQN=False):
    """
    This function calculates the friction factor for a pipe based on the
    notion of laminar, turbulent and transitional flow.
    :param Re: the Reynolds number under question.
    :param rr: the relative pipe roughness (expect between 0 and 0.05)
    :param CBEQN:  boolean to indicate if I should use Colebrook (True) or laminar equation
    :return: the (Darcy) friction factor
    """
    if CBEQN:
        # Implement the Colebrook equation as a lambda function with added safeguards
        def colebrook_eq(f, Re, rr):
            return 1 / np.sqrt(f) + 2.0 * np.log10((rr / 3.7) + (2.51 / (Re * np.sqrt(f))))  # Colebrook equation

        # Solve using fsolve with a safe initial guess
        result = fsolve(colebrook_eq, 0.02, args=(Re, rr))

        # Return the first solution (fsolve returns an array)
        return result[0] if result[0] > 0 else 0.02  # Ensure f > 0 for physical meaning
    else:
        return 64 / Re  # Laminar flow equation: f = 64/Re for Re < 2000


def plotMoody(plotPoint=False, pt=(0, 0)):
    """
    This function produces the Moody diagram for a Re range from 1 to 10^8 and
    for relative roughness from 0 to 0.05 (20 steps). The laminar region is described
    by the simple relationship of f=64/Re whereas the turbulent region is described by
    the Colebrook equation.
    :return: just shows the plot, nothing returned
    """
    # Step 1: Create logspace arrays for ranges of Re
    ReValsCB = np.logspace(np.log10(4000), np.log10(1E8),
                           100)  # for use with Colebrook equation (i.e., Re in range from 4000 to 10^8)
    ReValsL = np.logspace(np.log10(600.0), np.log10(2000.0),
                          20)  # for use with Laminar flow (i.e., Re in range from 600 to 2000)

    # Step 2: Create array for range of relative roughnesses
    rrVals = np.array(
        [0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4, 1E-3, 2E-3, 4E-3, 6E-3, 8E-8, 1.5E-2, 2E-2, 3E-2,
         4E-2, 5E-2])

    # Step 2: Calculate the friction factor in the laminar range
    ffLam = np.array([ff(Re, 0) for Re in ReValsL])  # Laminar flow: No roughness needed for laminar flow

    # Step 3: Calculate friction factor values for each rr at each Re for turbulent range.
    ffCB = np.array([[ff(Re, rr, CBEQN=True) for Re in ReValsCB] for rr in rrVals])

    # Step 4: Construct the plot
    plt.loglog(ReValsL, ffLam, label="Laminar", linestyle='-', color='b')  # plot the laminar part as a solid line

    # Plot the turbulent region for each relative roughness value
    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], color='k', label=f"Roughness {rrVals[nRelR]:.2e}")
        plt.annotate(f'{rrVals[nRelR]:.2e}', xy=(ReValsCB[-1], ffCB[nRelR, -1]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)

    plt.xlim(600, 1E8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number", fontsize=16)
    plt.ylabel(r"Friction factor", fontsize=16)
    plt.text(2.5E8, 0.02, r"Relative roughness $\frac{\epsilon}{d}$", rotation=90, fontsize=16)

    # Customize axes and ticks
    ax = plt.gca()  # capture the current axes for use in modifying ticks, grids, etc.
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)  # format tick marks
    ax.tick_params(axis='both', grid_linewidth=1, grid_linestyle='solid', grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')

    if plotPoint:
        plt.plot(pt[0], pt[1], 'ro', markersize=12, markeredgecolor='red', markerfacecolor='none')

    plt.show()


def main():
    plotMoody()


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
