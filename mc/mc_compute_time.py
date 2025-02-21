time_per_trajectory = 37  # in seconds
n_trajectory = 252 * 4 * 200
n_core = 800

total_time = time_per_trajectory * n_trajectory / n_core

# Beautiful print of the total time
minutes = int(total_time // 60)
seconds = total_time % 60

hours = int(minutes // 60)
minutes = minutes % 60


traj_per_core = n_trajectory // n_core
if n_trajectory % n_core > 0:
    traj_per_core = traj_per_core + 1

print(f"Trajectories per core: {traj_per_core}")
if hours > 0:
    print(f"Total time required: {hours} hour(s) and {minutes} minute(s) and {seconds:.2f} second(s)")
else:
    print(f"Total time required: {minutes} minute(s) and {seconds:.2f} second(s)")
