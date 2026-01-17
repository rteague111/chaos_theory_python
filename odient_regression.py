from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import sys

logging.basicConfig(
	filename="model_debug.log",
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(message)s"
)

#logging.info("model training started")
#logging.warning("Small numerical drift detected")
#logging.error("Prediction blew up")

logging.basicConfig(
    filename="model_debug.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Add this right after your logging setup to catch ALL unhandled exceptions
def log_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_unhandled_exception


#harmonic oscillation differental equations

#define system of ODE

def harmonic_oscillator (y, t, k, m):
	x, v = y #unpacks position and velocity
	dxdt = v
	dvdt = -(k/m)*x #Hooke's law F = -kx
	return [dxdt, dvdt]


#define parameters
k= 1.0 #spring constant
m = 1.0 #mass
y0 = [1.0, 0.0] #initial conditions x= 1, v=0

# time for array solution

t = np.linspace(0, 10, 100)

# solve ODE

solution = odeint(harmonic_oscillator, y0, t, args = (k, m))

# extract position and velocity
x, v = solution.T # transpose for easier unpacking

plt.figure(figsize=(10, 5))
plt.plot (t, x, label="Displacement (x)")
plt.plot (t, v, label="Velocity (v)", linestyle="dashed")
plt.xlabel("Time")

plt.legend()
plt.show()

# leaning towards state evolution learning; going to implement ML in ver 2
# bridge to control theory with eigenvalues
# generalize to higher dimensions - matriculation? extrapolation? abstraction

#adding data to dataframe
df = pd.DataFrame({
	
    "x": x[:-1],
	"v": v[:-1],
	"x_next": x[1:],
	"v_next": v[1:]
})

#linear regression segment from sklearn.linear_model
X = df[["x", "v"]]
Y = df[["x_next", "v_next"]]

#NOTE trained model for: [x(t), v(t)] → [x(t+Δt), v(t+Δt)]



model = LinearRegression()
model.fit(X, Y)
#adding noise
X_noisy = X + np.random.normal(0,0.01, size=X.shape)

M_learned = model.coef_
b_learned = model.intercept_
#this is state spaced learning


print("Learned transition matrix:\n", M_learned)
print("Learned bias:\n", b_learned)


#the following is one-step prediction
s0 = np.array([[x[0], v[0]]]) #initial state
s1_pred = model.predict(s0)

print("True next state:", [x[1], v[1]])
print("Predicted next state:", s1_pred[0])


# ADDING A ML MODEL INTO THE CODE!!

steps = 100
s = np.zeros((steps, 2))
s[0] = [x[0], v[0]]

for i in range(steps -1):
	try:
		s[i+1] = model.predict(s[i].reshape(1, -1))
	except Exception as e:
		logging.exception("Prediction failed at step %d", i)
		break


# compared to truth model

plt.plot(t[:steps], x[:steps], label="True x")
plt.plot(t[:steps], s[:, 0], "--", label="ML x")
plt.legend()
plt.savefig("ml_vs_truth.png")

# saves plt comparison for model validation

