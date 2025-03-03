# Import required libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the system of ODEs
def ode_system(t, X, A, Cd, ps, pa, V, beta, rho, Kvalve, m, y):
    """
    Defines the system of ODEs based on the given differential equations.

    State variables:
    X[0] = x (position)
    X[1] = xdot (velocity)
    X[2] = p1 (pressure on right side)
    X[3] = p2 (pressure on left side)
    """
    # Unpack state variables
    x, xdot, p1, p2 = X

    # Compute derivatives
    xddot = (p1 - p2) * A / m
    p1dot = (y * Kvalve * (ps - p1) - rho * A * xdot) * beta / V
    p2dot = (y * Kvalve * (p2 - pa) - rho * A * xdot) * beta / V

    return [xdot, xddot, p1dot, p2dot]


# Define main function to solve and plot results
def main():
    # Time range
    t_span = [0, 0.02]  # Simulating for 20 ms
    t_eval = np.linspace(0, 0.02, 200)

    # Given parameters
    A = 4.909E-4
    Cd = 0.6
    ps = 1.4E7
    pa = 1.0E5
    V = 1.473E-4
    beta = 2.0E9
    rho = 850.0
    Kvalve = 2.0E-5
    m = 30
    y = 0.002  # Constant input

    # Initial conditions: [x, xdot, p1, p2]
    ic = [0, 0, pa, pa]

    # Solve the ODEs
    sol = solve_ivp(ode_system, t_span, ic, args=(A, Cd, ps, pa, V, beta, rho, Kvalve, m, y), t_eval=t_eval)

    # Extract results
    t = sol.t
    xvals = sol.y[0]
    xdot = sol.y[1]
    p1 = sol.y[2]
    p2 = sol.y[3]

    # Plot x vs. time
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, xvals, 'r-', label='$x$ (Position)')
    plt.ylabel('Position $x$ (m)')
    plt.legend(loc='upper left')

    ax2 = plt.twinx()
    ax2.plot(t, xdot, 'b-', label=r'$\dot{x}$ (Velocity)')
    ax2.set_ylabel(r'Velocity $\dot{x}$ (m/s)')
    ax2.legend(loc='lower right')

    # Plot p1 and p2 vs. time
    plt.subplot(2, 1, 2)
    plt.plot(t, p1, 'b-', label='$P_1$')
    plt.plot(t, p2, 'r-', label='$P_2$')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


# Execute main function
if __name__ == "__main__":
    main()
