import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def f_x(x, y, a, K, b):
    return a * x * (K - x) / K - b * x * y

def f_y(x, y, e, c, d):
    return -e * y**2 - c * y + d * x * y

def step(x_n, y_n, h, a, K, b, e, c, d):
    def lokta(vars):
        x_next, y_next = vars
        eq1 = x_next - x_n - (h / 2) * (f_x(x_n, y_n, a, K, b) + f_x(x_next, y_next, a, K, b))
        eq2 = y_next - y_n - (h / 2) * (f_y(x_n, y_n, e, c, d) + f_y(x_next, y_next, e, c, d))
        return [eq1, eq2]
    
    x_next, y_next = opt.fsolve(lokta, (x_n, y_n))
    return x_next, y_next

def implicit_solve(x0, y0, t0, t_end, h, a, K, b, e, c, d):
    t_v = np.arange(t0, t_end + h, h)
    x_v = np.zeros(len(t_v))
    y_v = np.zeros(len(t_v))
    
    x_v[0] = x0
    y_v[0] = y0
    
    for i in range(1, len(t_v)):
        x_v[i], y_v[i] = step(x_v[i-1], y_v[i-1], h, a, K, b, e, c, d)
    
    return t_v, x_v, y_v

def plot_results(t_v, x_v, y_v):
    plt.figure(figsize=(10, 5))
    plt.plot(t_v, x_v, label="Fisherman")
    plt.plot(t_v, y_v, label="Fish")
    plt.title('Implicit solution')
    plt.legend()
    plt.show()

# Parameters
# Fish
a, K, b = 1.0, 10, 0.5
# Fisherman
e, c, d = 0.01, 0.2, 0.2
#Initial values
x0, y0 = 2.0, 3.0
t0, t_end, h = 0, 50, 0.1

t_v, x_v, y_v = implicit_solve(x0, y0, t0, t_end, h, a, K, b, e, c, d)

plot_results(t_v, x_v, y_v)
