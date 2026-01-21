import matplotlib.pyplot as plt

# Data
minutes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
H = [0.2388, 0.2319, 0.2071, 0.1941, 0.1948, 0.2128, 0.213, 0.2078, 0.1995, 0.1967, 0.1996]

# Plot
plt.figure()
plt.plot(minutes, H, marker='o')
plt.xlabel("Window (minutes)")
plt.ylabel("H value")
plt.title("H vs Time")
plt.ylim(0.1, 0.3)
plt.show()
