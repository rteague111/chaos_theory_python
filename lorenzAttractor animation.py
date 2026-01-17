import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Extract solution
x, y, z = solution.y

#skipping frames
skip = 10
x = x[::skip]
y = y[::skip]
z = z[::skip]


# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim((min(x), max(x)))
ax.set_ylim((min(y), max(y)))
ax.set_zlim((min(z), max(z)))
ax.set_title("Lorenz Attractor Animation")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Plot line and set up animation
line, = ax.plot([], [], [], lw=0.5)

# Animation function
def update(num):
    line.set_data(x[:num], y[:num])
    line.set_3d_properties(z[:num])
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)

plt.show()