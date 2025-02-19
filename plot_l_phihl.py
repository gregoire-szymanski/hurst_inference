import numpy as np
import matplotlib.pyplot as plt
from estimator_H import Phi_Hl

# Parameters
H_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49]
l_values = np.linspace(2, 10, 81)  # Range of l values

# Initialize plot
plt.figure(figsize=(10, 6))

for H in H_values:
    # Precompute normalizing factor
    zero = Phi_Hl(0, H) + 2 * Phi_Hl(1, H)
    
    # Compute Phi values
    Phi_values = np.array([Phi_Hl(l, H) / zero for l in l_values])
    
    # Plot each curve
    plt.plot(l_values, Phi_values, label=rf"$H = {H}$", linewidth=2)

# Titles and labels
plt.title(r"$\Phi^H_\ell$ for Different Values of $H$", fontsize=14)
plt.xlabel(r"$\ell$", fontsize=12)
plt.ylabel(r"$\Phi^H_\ell$", fontsize=12)

# Styling
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
# plt.xlim([min(l_values), max(l_values)])
# plt.ylim([0, max(max(Phi_values) for H in H_values)])  # Ensure proper y-axis range

# Show plot
plt.show()
