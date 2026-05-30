import numpy as np 
import matplotlib.pyplot as plt
from grad_exp import var,fnmatmul, fnsigmoid, autograd 

class layer():
    
    def __init__(self,numnodesin,numnodesout):
        self.numnodesin=numnodesin
        self.numnodesout=numnodesout
        self.weights = var(2 * np.random.random((numnodesout, numnodesin)) - 1)#weights and biases have initial value between -1 and 1
        self.biases = var(2 * np.random.random((numnodesout, 1)) - 1)
        self.mw= np.zeros_like(self.weights.val)#initialise momentum for weights
        self.vw= np.zeros_like(self.weights.val)#initialise velocity for weights
        self.mb = np.zeros_like(self.biases.val)#same for biases
        self.vb = np.zeros_like(self.biases.val)
        self.t=0

    # #older optimiser
    # def applygrad(self,lr,grads):
    #     if self.weights in grads:
    #         self.weights.val-=lr*grads[self.weights].val
    #     if self.biases in grads:
    #         self.biases.val-=lr*grads[self.biases].val

    def adam(self,grads,lr,beta1,beta2,ep):
        self.t+=1
        self.mw=beta1*self.mw+(1-beta1)*grads[self.weights].val
        self.vw=beta2*self.vw+(1-beta2)*(grads[self.weights].val*grads[self.weights].val)
        mwhat=self.mw/(1-beta1**self.t)
        vwhat=self.vw/(1-beta2**self.t)
        self.weights.val-=lr*mwhat/(np.sqrt(vwhat)+ep)
        self.mb=beta1*self.mb+(1-beta1)*grads[self.biases].val
        self.vb=beta2*self.vb+(1-beta2)*(grads[self.biases].val*grads[self.biases].val)
        mbhat=self.mb/(1-beta1**self.t)
        vbhat=self.vb/(1-beta2**self.t)
        self.biases.val-=lr*mbhat/(np.sqrt(vbhat)+ep)

    def calcoutputs(self, inputs,is_output=False):
        if not isinstance(inputs, var):
            inputs = var(inputs)
        z = fnmatmul(self.weights, inputs) + self.biases#Z=W*I+B
        if is_output:
            return z#no activation on final layer so that output can be unbounded  
        return fnsigmoid(z)#A=sig(Z)


class nn():
    def __init__(self,layersizes):
        self.layersizes=layersizes
        self.layers=[]
        for i in range(len(layersizes)-1):
            self.layers.append(layer(self.layersizes[i],self.layersizes[i+1]))

    def calc(self, inputs):
        for i, x in enumerate(self.layers):
            is_output = (i == len(self.layers) - 1)#this check so that we can not apply activation in output layer in calcoutputs
            inputs = x.calcoutputs(inputs, is_output=is_output)
        return inputs
    
    def learn_pinn(self,epochs,lr):
        for e in range(epochs):
            phyloss = var(np.zeros((1,1)))#initialise to [[0]]
            for t in t_collocation:#t_collocation is list of our our training points
                t_var=var(np.array([[t]]))#making the point to a var object so that its compatible with out network 
                u= self.calc(t_var)#forward pass
                dudt=autograd(u)[t_var]
                residual=dudt + u
                phyloss+=residual ** 2
            #boundary condition u(0)=1
            t0=var(np.array([[0.0]]))
            u0 =self.calc(t0)
            boundary_loss=(u0-var(np.array([[1.0]])))**2
            total_loss = phyloss + var(np.array(50.0)) * boundary_loss  # weight boundary more
            #update weights
            grads=autograd(total_loss)
            for layer in self.layers:
                layer.adam(grads,lr,0.9,0.999,1e-8)
            
            if e % 1000 == 0:
                print(f"epoch {e}  loss: {total_loss.val[0][0]:.4f}")
        print(f"Final loss: {total_loss.val[0][0]:.4f}")


#ode is du/dt=-u and boundary point is u(0)=1
np.random.seed(54)
pinn1 = nn([1, 20, 20, 1])


t_collocation=np.linspace(0, 2, 100)#100 points from t=0 to 2
pinn1 = nn([1, 20, 20, 1])
pinn1.learn_pinn(epochs=10000, lr=0.001)
# compare against analytical solution u(t) = e^(-t)
t_test=np.linspace(0, 3, 300)
u_analytical=np.exp(-t_test)
u_pinn=[pinn1.calc(var(np.array([[t]]))).val[0][0] for t in t_test]

plt.plot(t_test, u_analytical, label='Analytical:e^(-t)', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()
