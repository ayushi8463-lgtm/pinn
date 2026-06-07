import numpy as np
import time

steps= [100, 300, 1000, 3000, 10000]
t_analytical = np.linspace(0, 3, 300)
u_analytical = np.exp(t_analytical)
# Solves d4u/dt4 = u
def f(s):#s = [u, u', u'', u''']
    return np.array([s[1], s[2], s[3], s[0]])

def rk4(t_span, y0, n_steps):
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)

    y = np.zeros((n_steps + 1, 4))
    y[0] = y0

    for i in range(n_steps):
        k1 = f(y[i])
        k2 = f(y[i] + h*k1/2)
        k3 = f(y[i] + h*k2/2)
        k4 = f(y[i] + h*k3)
        y[i+1] = y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t, y[:, 0]

# u(0)=1, u'(0)=1, u''(0)=1, u'''(0)=1 
y0 = [1.0, 1.0, 1.0, 1.0]

for n in steps:
    start=time.time()
    t, u_rk4 = rk4((0, 3), y0, n)
    elapsed=time.time()-start

    u_interp = np.interp(t_analytical, t, u_rk4)

    l2 = np.sqrt(np.sum((u_interp - u_analytical)**2)) / \
         np.sqrt(np.sum(u_analytical**2))

    print(f"steps={n:6d}  L2={l2:.2e}  time={elapsed:.4f}s")