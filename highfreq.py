import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

W0 = 10.0
#initialisation
def initialise(layersizes,key):
    weights=[]
    biases=[]
    mw=[]
    vw=[]
    mb=[]
    vb=[]
    for i,(numnodesin,numnodesout) in enumerate(zip(layersizes[:-1],layersizes[1:])):
        key, subkey1,subkey2= jax.random.split(key,3)
        if i==0:
            limit= 1.0/numnodesin
        else:
            limit= jnp.sqrt(6.0 / numnodesin)/W0 
        w= jax.random.uniform(subkey1, (numnodesout, numnodesin), minval=-limit, maxval=limit)
        b_limit= 1.0 / numnodesin
        b=jax.random.uniform(subkey2, (numnodesout, 1), minval=-b_limit, maxval=b_limit)
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
    a=jnp.array([[t]])
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=w@a+b
        is_last= (i== len(biases)-1)
        if not is_last:
            a=jnp.sin(W0 * a)
            
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
    total_loss=phyloss+boundary_loss
    return total_loss

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
    t_collocation = jnp.linspace(0,3*jnp.pi,500)
    for e in range(epochs):
        weights, biases, mw, vw, mb, vb, ta, loss = adam(
            weights, biases, t_collocation, mw, vw, mb, vb, ta, lr)
        if e % 1000 == 0:
            print(f"epoch {e}  loss: {loss:.6f}")
    print(f"epoch {e}  loss: {loss:.6f}")  
    return weights, biases

start=time.time()
key = jax.random.PRNGKey(50)
weights,biases=learn([1, 64, 1], 15000, 0.001,key)
weights[0].block_until_ready()
elapsed=time.time()-start
t_test=jnp.linspace(0,3*jnp.pi, 300)
u_analytical=jnp.cos(5*t_test)
ufn= lambda t: calc(t, weights,biases)[0, 0]
u_pinn = jax.vmap(ufn)(t_test)
print(f"time taken {elapsed:.4f}s")

plt.plot(t_test, u_analytical, label='Analytical: cos(5t)', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.show()
