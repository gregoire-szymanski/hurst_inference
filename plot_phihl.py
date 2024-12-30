import numpy as np
import matplotlib.pyplot as plt

def Phi_Hl(l, H):
    """
    Compute the value of Phi^H_ell based on the given formula.

    :param l: The parameter ell in the formula
    :param H: The parameter H (Hurst exponent) in the formula
    :return: The computed value of Phi^H_ell
    """
    numerator = (np.abs(l + 2) ** (2 * H + 2) - 4 * np.abs(l + 1) ** (2 * H + 2) +
                 6 * np.abs(l) ** (2 * H + 2) - 4 * np.abs(l - 1) ** (2 * H + 2) +
                 np.abs(l - 2) ** (2 * H + 2))
    denominator = 2 * (2 * H + 1) * (2 * H + 2)
    return numerator / denominator

# Parameters
H_values = [0.01, .05, .1, .2, .3, .4, 0.45, .49]
l_values = np.arange(0, 11)  # Range of l values

# # Plotting
# plt.figure(figsize=(10, 6))

# for H in H_values:
#     Phi_values = np.array([Phi_Hl(l, H) for l in l_values])
#     plt.plot(l_values, Phi_values / Phi_values[0], label=f"H = {H}")

# plt.title("Phi^H_ell for Different Values of H")
# plt.xlabel("l")
# plt.ylabel("Phi^H_ell")
# plt.legend()
# plt.grid(True)
# plt.show()


H_values = np.linspace(0.001, 0.999, 1000)

phi0 = np.array([Phi_Hl(0, H) for H in H_values])
phi2 = np.array([Phi_Hl(2, H) for H in H_values])
phi3 = np.array([Phi_Hl(3, H) for H in H_values])
phi4 = np.array([Phi_Hl(4, H) for H in H_values])



# Plotting
plt.figure(figsize=(10, 6))

plt.plot(H_values, phi2 / phi0)
plt.plot(H_values, phi3 / phi0)
plt.plot(H_values, phi4 / phi0)

plt.title("Phi^H_ell for Different Values of H")
plt.xlabel("H")
plt.ylabel("Phi^H_ell")
plt.legend()
plt.grid(True)
plt.show()
