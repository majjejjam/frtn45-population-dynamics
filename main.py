import numpy as np
import matplotlib.pyplot as plt
import scipy

#General RK4 method
def rk4(f, told, uold, h):
  c = np.array([0, 1/2, 1/2, 1])
  A = np.array([[0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1, 0]])
  b = np.array([1/6, 1/3, 1/3, 1/6])

  stage_derivatives = []
  for i in range(4):
    if i == 0:
      stage_derivatives.append(f(told, uold))
    else:
      stage_derivatives.append(f(told + c[i]*h, uold + h*A[i,i-1]*stage_derivatives[i-1]))

  unew = uold
  for i in range(4):
    unew += h*b[i]*stage_derivatives[i]

  return unew

#RK34 method
def rk34(f, told, uold, h):
  c = np.array([0, 1/2, 1/2, 1])
  A = np.array([[0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1, 0]])
  b = np.array([1/6, 1/3, 1/3, 1/6])

  z = np.array([1/6,2/3,1/6])
  stage_derivatives = []
  for i in range(4):
    if i == 0:
      stage_derivatives.append(f(told, uold))
    else:
      stage_derivatives.append(f(told + c[i]*h, uold + h*A[i,i-1]*stage_derivatives[i-1]))

  Z_prime = f(told + h, uold -h*stage_derivatives[0]+2*h*stage_derivatives[1])
  unew = uold
  for i in range(4):
    unew += h*b[i]*stage_derivatives[i]

  lnew = h/6*(2*stage_derivatives[1]+Z_prime-2*stage_derivatives[2]-stage_derivatives[3])

  return unew,lnew

# Task 1.3
def newstep(tol, err, errold, hold, k):
    # Compute the new step size using the given formula
    r_n = err
    r_nold = errold
    hnew = (tol / r_n)**(2/(3 * k)) * (tol / r_nold)**(-1/(3 * k)) * hold

    return hnew

# Exempel
tol = 1e-7  # Tolerance
err = 0.02  # Current error estimate
errold = 0.03  # Previous error estimate
hold = 0.1  # Previous step size
k = 4  # Order of the error estimator (e.g., for RK4)

hnew = newstep(tol, err, errold, hold, k)
print(f"New step size: {hnew}")

# Task 1.4
def adaptiveRK34(f, t0, tf, y0, tol):
  t_vec = [t0]  # Initialize time vector with t0
  u_vec = [y0]  # Initialize solution vector with y0
  print("This is startvalue: ", u_vec)
  t = t0
  errold = tol  # Initial previous error estimate
  h = (np.abs(tf - t0) * (tol**(1 / 4))) / (100 * (1 + np.linalg.norm(f(t0, y0))))
  uold = y0
  k = 4  # Order of the RK4 method

  while t_vec[-1] < tf:

      t = t_vec[-1]
      uold = u_vec[-1]
      # Adjust step size to avoid overshooting tf
      if t + h > tf:
          h = tf - t

      # Compute new solution and error estimate using RK34
      unew, err = rk34(f, t, uold, h)

      t += h
      t_vec.append(t)
      u_vec.append(np.copy(unew))

      errold = np.linalg.norm(err)
      # Update step size using newstep
      hold = h
      h = newstep(tol, np.linalg.norm(err), errold, hold, k)
  return np.array(t_vec), np.array(u_vec)


def lotka_volterra(t, u):
    x, y = u
    a, b, c, d = 3, 9, 15, 15

    dxdt = a * x - b * x * y
    dydt = c * x * y - d * y

    return np.array([dxdt, dydt])


# Initial conditions
t0 = 0
tf = 50
y0 = np.array([2, 1], dtype=np.float64)  # Ensure the initial condition is float64

tol = 1e-6  # Tolerance

# Run adaptive RK34 solver
t_vec, u_vec = adaptiveRK34(lotka_volterra, t0, tf, y0, tol)

# Extract solutions
u_vec = np.array(u_vec)

x = u_vec[:, 0]  # Prey
y = u_vec[:, 1]  # Predator

# Plot population dynamics
plt.figure(figsize=(12, 4))
plt.plot(t_vec, x, label="Prey (x)", color="blue")
plt.plot(t_vec, y, label="Predator (y)", color="orange")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka-Volterra: Population Dynamics (Adaptive RK34)")
plt.legend()
plt.grid()
plt.show()

print("This is x0,y0: ", x[0], " ", y[0])
print("This is x1,y1: ", x[1], " ", y[1])
# Phase portrait
X = np.linspace(0, 4, 20)
Y = np.linspace(0, 4, 20)
U,V = np.meshgrid(X, Y)
u, v = lotka_volterra(0, np.array([U, V]))

dxdt, dydt = lotka_volterra(0, np.array([X, Y]))
plt.figure(figsize=(10, 6))
plt.streamplot(U,V,u,v)
plt.plot(x, y, color="green")
plt.xlabel("Prey (x)")
plt.ylabel("Predator (y)")
plt.title("Phase Portrait: Predator-Prey")
plt.grid()
plt.show()


# Initial conditions
t0 = 0
tf = 1000
y0 = np.array([2, 1], dtype=np.float64)  # Ensure the initial condition is float64

#y0 = np.array([1, 1])  # Initial populations
tol = 1e-8  # Tolerance

# Run adaptive RK34 solver
t_vec, u_vec = adaptiveRK34(lotka_volterra, t0, tf, y0, tol)

# Extract solutions
u_vec = np.array(u_vec)
t_vec = np.array(t_vec)

x = u_vec[:, 0]  # Prey
y = u_vec[:, 1]

a,b,c,d = 3,9,15,15
H = lambda x, y: c * x - d * np.log(x) + b * y - a * np.log(y)
H0 = H(x[0], y[0])

deviation = np.abs(H(x, y) / H0 - 1)

# Plot deviation
plt.figure(figsize=(10, 6))
plt.plot(t_vec, deviation, label=r"$|H(x, y)/H(x(0), y(0)) - 1|$")
plt.xlabel("Time (t)")
plt.ylabel("Normalized Deviation")
plt.title("Deviation Over Time")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()