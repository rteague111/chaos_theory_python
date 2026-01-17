#from chatGPT

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Lorenz system equations
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]  # Initial conditions
t_span = (0, 50)  # Time range
t_eval = np.linspace(*t_span, 10000)  # Time points for evaluation

# Solve the system
solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# Plot the attractor
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(solution.y[0], solution.y[1], solution.y[2], lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()