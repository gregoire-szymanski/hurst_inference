import numpy as np
import matplotlib.pyplot as plt

from lib.estimator_H import Phi_Hl

H_values = np.linspace(0.001, 0.999, 1000)
l_values = [0, 1, 2, 3, 4]
norm = False

phi0 = np.array([Phi_Hl(0, H) for H in H_values])
phi_l = [
    np.array([Phi_Hl(l, H) for H in H_values]) 
    for l in l_values
]

# Apply normalization correctly
if norm:
    phi_l = [phi / phi0 for phi in phi_l]

# Plotting
plt.figure(figsize=(10, 6))

for l, phi in zip(l_values, phi_l):
    plt.plot(H_values, phi, label=f"$\ell = {l}$", linewidth=2)

plt.title(r"$\Phi^H_\ell$ for Different Values of $\ell$", fontsize=14)
plt.xlabel(r"$H$", fontsize=12)
plt.ylabel(r"$\Phi^H_\ell$", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim([0, 1])
plt.ylim([min(min(phi) for phi in phi_l), max(max(phi) for phi in phi_l)])

plt.show()
