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
        key, subkey= jax.random.split(key)
        limit = jnp.sqrt(6.0 / (numnodesin + numnodesout))#xavier intialisation
        w = jax.random.uniform(subkey, (numnodesout, numnodesin), minval=-limit, maxval=limit)
        b = jnp.zeros((numnodesout, 1))

        weights.append(w)
        biases.append(b)
        mw.append(jnp.zeros_like(w))
        vw.append(jnp.zeros_like(w))
        mb.append(jnp.zeros_like(b))
        vb.append(jnp.zeros_like(b))
    ta=jnp.zeros((),dtype=jnp.int32)
    return weights,biases,mw,vw,mb,vb,ta

#forward pass
def calc(t,weights,biases):
    a=jnp.array([[t]])#convert scalar t into 1x1 matrix
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=w@a+b
        is_last = (i == len(biases)-1)
        if not is_last:
            a = jnp.tanh(a) 
    return a

def calcloss(weights, biases, t_collocation):
    ufn= lambda t: calc(t, weights, biases)[0, 0]
    dudtfn= jax.grad(ufn)
    d2udt2fn= jax.grad(dudtfn)
    d3udt3fn= jax.grad(d2udt2fn)
    d4udt4fn= jax.grad(d3udt3fn)

    u = jax.vmap(ufn)(t_collocation)
    d4udt4= jax.vmap(d4udt4fn)(t_collocation)

    residual= d4udt4 - u
    phyloss= jnp.mean(residual**2)

    # u(0)=1, u'(0)=1, u(1)=e, u'(1)=e  (from u=e^x)
    u0= ufn(0.0)
    dudt0= dudtfn(0.0)
    d2udt20 = d2udt2fn(0.0)
    d3udt30 = d3udt3fn(0.0)
    boundary_loss = (u0 - 1.0)**2 + (dudt0 - 1.0)**2 + (d2udt20- 1)**2 + (d3udt30 - 1)**2

    return phyloss + boundary_loss

@jax.jit #to speed up subsequent calls
def adam(weights, biases, t_collocation, mw, vw, mb, vb, ta, lr):
    loss,(grad_w,grad_b)=jax.value_and_grad(calcloss,argnums=(0,1))(weights,biases,t_collocation)
    ta=ta+1
    beta1=0.9 
    beta2=0.999
    ep=1e-8

    mw=[beta1*m+ (1-beta1)*g for m,g in zip(mw, grad_w)]
    vw=[beta2*v+ (1-beta2)*g**2 for v,g in zip(vw, grad_w)]
    mb=[beta1*m+ (1-beta1)*g for m, g in zip(mb, grad_b)]
    vb=[beta2*v+ (1-beta2)*g**2 for v, g in zip(vb, grad_b)]

    mwhat= [m/ (1- beta1**ta) for m in mw]
    vwhat= [v/ (1 - beta2**ta) for v in vw]
    mbhat= [m/ (1 - beta1**ta) for m in mb]
    vbhat= [v/ (1 - beta2**ta) for v in vb]

    effective_lr = jnp.where(ta >= 10000, lr * 0.1, lr)
    new_weights = [w - effective_lr * m / (jnp.sqrt(v) + ep) for w, m, v in zip(weights, mwhat, vwhat)]
    new_biases  = [b - effective_lr * m / (jnp.sqrt(v) + ep) for b, m, v in zip(biases,  mbhat, vbhat)]


    return new_weights, new_biases, mw, vw, mb, vb,ta, loss

def learn(layersizes,epochs,lr,key):
    weights,biases,mw,vw,mb,vb,ta=initialise(layersizes,key)
    t_collocation = jnp.linspace(0,3, 500)
    for e in range(epochs):
        weights, biases, mw, vw, mb, vb, ta, loss = adam(
            weights, biases, t_collocation, mw, vw, mb, vb, ta, lr)
        if e % 1000 == 0:
            print(f"epoch {e}  loss: {loss:.6f}")
    print(f"epoch {e}  loss: {loss:.6f}")  
    return weights, biases

start=time.time()
key = jax.random.PRNGKey(50)
weights,biases=learn([1,64,64,64, 1], 15000, 0.001,key)
weights[0].block_until_ready()
elapsed=time.time()-start
t_test=jnp.linspace(0,3, 300)
u_analytical =jnp.exp(t_test)
ufn= lambda t: calc(t, weights,biases)[0, 0]
u_pinn = jax.vmap(ufn)(t_test)
print(f"time taken {elapsed:.4f}s")

plt.plot(t_test, u_analytical, label='Analytical: e^t', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.show()
