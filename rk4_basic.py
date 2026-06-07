import numpy as np
import time

steps=[100, 300, 1000, 3000, 10000]
t_test=np.linspace(0, 3*np.pi, 300)
u_analytical=np.cos(t_test)
#Solves d2u/dt2 = -u

def rk4(t_span, u0, dudt0, n_steps):
    t0, tf = t_span
    h=(tf - t0) / n_steps
    t=np.linspace(t0, tf, n_steps + 1)
    
    y=np.zeros((n_steps + 1, 2))#a vector with number of rows=number of points and 2 values at each point
    y[0]=[u0, dudt0]
    
    def f(s):#s=[u ,du/dt ]
        return np.array([s[1], -s[0]]) #f(s)=[du/dt,d2u/dt2]=[du/dt,-25u]
    
    for i in range(n_steps):
        k1= f(y[i])
        k2= f(y[i] + h*k1/2)
        k3= f(y[i] + h*k2/2)
        k4= f(y[i] + h*k3)
        y[i+1]=y[i]+(h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return t,y[:, 0]#return u at all points

for n in steps:
    start=time.time()
    t,urk4=rk4((0, 3*np.pi), 1.0, 0.0, n)#at x=0 u=1 du/dt=0
    elapsed=time.time()-start
    
    #interpolate to same points as analytical
    u_interp = np.interp(t_test,t, urk4)
    
    #relative l2 error
    l2 = np.sqrt(np.sum((u_interp - u_analytical)**2)) /np.sqrt(np.sum(u_analytical**2))

    print(f"steps={n:6d}  L2={l2:.2e}  time={elapsed:.4f}s")

    

