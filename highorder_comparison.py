import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#initialisation
def initialise(layersizes,key):
    weights=[]
    biases=[]
    #adam initialisation
    mw=[]
    vw=[]
    mb=[]
    vb=[]
    for (numnodesin,numnodesout) in zip(layersizes[:-1],layersizes[1:]):
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
    t=0
    return weights,biases,mw,vw,mb,vb,t

#forward pass
def calc(t,weights,biases):
    a=jnp.array([[t]])#convert scalar t into 1x1 matrix
    for i,(w,b) in enumerate(zip(weights,biases)):
        a=w@a+b
        is_last = (i == len(biases)-1)
        if not is_last:
            a = activation(a) 
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
    t_collocation = jnp.linspace(0,3, 500)
    for e in range(epochs):
        current_lr = lr if e < 10000 else lr * 0.1
        weights, biases, mw, vw, mb, vb, t, loss = adam(
            weights, biases, t_collocation, mw, vw, mb, vb, t, current_lr)
        # if e % 1000 == 0:
        #     print(f"epoch {e}  loss: {loss:.6f}")
    return weights, biases

def test(pinn):
    t_test=jnp.linspace(0,3, 300)
    u_analytical=jnp.exp(t_test)
    u_pinn=jnp.array([calc(t,*pinn)[0,0] for t in t_test])
    l2_error = jnp.sqrt(jnp.sum((u_pinn - u_analytical)**2)) / jnp.sqrt(jnp.sum(u_analytical**2))
    return l2_error



exp=[[[1,16,16,1],[1,32,32,1],[1,64,64,1]],
     [[1,16,16,16,1],[1,32,32,32,1],[1,64,64,64,1]],
     [[1,16,16,16,16,1],[1,32,32,32,32,1],[1,64,64,64,64,1]]]

act=[jax.nn.sigmoid,jnp.tanh,jnp.sin]

print(f"{'Architecture':<22} {'Median L2':<12} {'Std':<12} {'Min':<12} {'Max':<12}")


for activation in act:
    adam.clear_cache()
    print(f"Activation function: {activation.__name__}")
    for i in exp:
        for j in i:
            errors = []
            for seed in [0,10,20,30,40]:
                key = jax.random.PRNGKey(seed)
                pinn=learn(j, 15000, 0.001,key)
                loss=test(pinn)
                errors.append(loss)
            errors=sorted([float(e) for e in errors])
            median = errors[len(errors) // 2] if len(errors) % 2 != 0 else (errors[len(errors)//2 - 1] + errors[len(errors)//2]) / 2
            std= (sum((e - median)**2 for e in errors) / len(errors)) ** 0.5
            mn= min(errors)
            mx = max(errors)
            print(f"{str(j):<22} {median:<12.4f} {std:<12.4f} {mn:<12.4f} {mx:<12.4f}")




