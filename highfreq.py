import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

#initialisation
def initialise(layersizes,key):
    weights=[]
    biases=[]
    #adam initialisation
    mw=[]
    vw=[]
    mb=[]
    vb=[]
    for i,(numnodesin,numnodesout) in enumerate(zip(layersizes[:-1],layersizes[1:])):
        key, subkey1,subkey2= jax.random.split(key,3)
        if i==0:
            limit = 1.0 / numnodesin
        else:
            limit = jnp.sqrt(6.0 / numnodesin) 
        w = jax.random.uniform(subkey1, (numnodesout, numnodesin), minval=-limit, maxval=limit)
        b_limit = 1.0 / numnodesin
        b = jax.random.uniform(subkey2, (numnodesout, 1), minval=-b_limit, maxval=b_limit)

        weights.append(w)
        biases.append(b)
        mw.append(jnp.zeros_like(w))
        vw.append(jnp.zeros_like(w))
        mb.append(jnp.zeros_like(b))
        vb.append(jnp.zeros_like(b))
    t=0
    return weights,biases,mw,vw,mb,vb,t

#forward pass
def calc(t,weights,biases):
    a=jnp.array([[t]])#convert scalar t into 1x1 matrix
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=w@a+b
        is_last = (i == len(biases)-1)
        if not is_last:
            a = jnp.sin(a) 
    return a

def calcloss(weights,biases,t_collocation):

    ufn=lambda t: calc(t,weights,biases)[0,0]#this takes single scalar point from t_collocation
    dudtfn=jax.grad(ufn)
    d2udt2fn=jax.grad(dudtfn)

    u=jax.vmap(ufn)(t_collocation)#returns array of size (batchsize,) after running ufn on all points
    d2udt2=jax.vmap(d2udt2fn)(t_collocation)
    residual= d2udt2+25*u
    phyloss=jnp.mean(residual**2)

    u0=ufn(0.0)
    dudt0=dudtfn(0.0)
    boundary_loss=(u0-1.0)**2+dudt0**2
    total_loss=phyloss+boundary_loss*500
    return total_loss

@jax.jit #to speed up subsequent calls
def adam(weights, biases, t_collocation, mw, vw, mb, vb, t, lr):
    loss,(grad_w,grad_b)=jax.value_and_grad(calcloss,argnums=(0,1))(weights,biases,t_collocation)
    t=t+1
    beta1=0.9 
    beta2=0.999
    ep=1e-8

    mw=[beta1*m+ (1-beta1)*g for m,g in zip(mw, grad_w)]
    vw=[beta2*v+ (1-beta2)*g**2 for v,g in zip(vw, grad_w)]
    mb=[beta1*m+ (1-beta1)*g for m, g in zip(mb, grad_b)]
    vb=[beta2*v+ (1-beta2)*g**2 for v, g in zip(vb, grad_b)]

    mwhat= [m/ (1- beta1**t) for m in mw]
    vwhat= [v/ (1 - beta2**t) for v in vw]
    mbhat= [m/ (1 - beta1**t) for m in mb]
    vbhat= [v/ (1 - beta2**t) for v in vb]

    new_weights = [w - lr * m / (jnp.sqrt(v) + ep) for w, m, v in zip(weights, mwhat, vwhat)]
    new_biases  = [b - lr * m / (jnp.sqrt(v) + ep) for b, m, v in zip(biases,  mbhat, vbhat)]


    return new_weights, new_biases, mw, vw, mb, vb,t, loss

def learn(layersizes,epochs,lr,key):
    weights,biases,mw,vw,mb,vb,t=initialise(layersizes,key)
    t_collocation = jnp.linspace(0,3*jnp.pi, 1000)
    for e in range(epochs):
        
        current_lr = lr if e < 10000 else lr * 0.1
        weights, biases, mw, vw, mb, vb, t, loss = adam(
            weights, biases, t_collocation, mw, vw, mb, vb, t, current_lr)
        if e % 1000 == 0:
            print(f"epoch {e}  loss: {loss:.6f}")
    print(f"epoch {e}  loss: {loss:.6f}")  
    return weights, biases

start=time.time()
key = jax.random.PRNGKey(50)
pinn1=learn([1, 64,64,64,64, 1], 15000, 0.001,key)
elapsed=time.time()-start
t_test=jnp.linspace(0,3*jnp.pi, 300)
u_analytical=jnp.cos(5*t_test)
u_pinn=[calc(t,*pinn1)[0,0] for t in t_test]
print(f"time taken {elapsed:.4f}s")

plt.plot(t_test, u_analytical, label='Analytical: cos(t)', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.show()
