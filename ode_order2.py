import numpy as np 
import matplotlib.pyplot as plt
from grad_exp import var,fnmatmul, fnsigmoid, autograd,fnsum_all,fntanh

class layer():
    
    def __init__(self,numnodesin,numnodesout):
        self.numnodesin=numnodesin
        self.numnodesout=numnodesout
        self.weights= var(2 * np.random.random((numnodesout, numnodesin)) - 1)#weights and biases have initial value between -1 and 1
        self.biases= var(2 * np.random.random((numnodesout, 1)) - 1)
        self.mw= np.zeros_like(self.weights.val)#initialise momentum for weights
        self.vw= np.zeros_like(self.weights.val)#initialise velocity for weights
        self.mb = np.zeros_like(self.biases.val)#same for biases
        self.vb = np.zeros_like(self.biases.val)
        self.t=0

    # #older optimiser
    # def applygrad(self,lr,grads):
    #     self.weights.val-=lr*grads[self.weights].val
    #     self.biases.val-=lr*grads[self.biases].val

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
        return fntanh(z)#A=tanh(Z)


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
    
    def learn_pinn(self, epochs, lr):
        for e in range(epochs):
            batch_size=len(t_collocation)
            t_batch = var(t_collocation.reshape(1, -1))
            u_batch = self.calc(t_batch)
            dudt = autograd(u_batch)[t_batch] #put it through autograd engine
            d2udt2=autograd(dudt)[t_batch]
            residual=d2udt2+u_batch #(1,len(t_collocation))
            phyloss = fnsum_all(residual ** 2)* var(1/batch_size)#add all and divide by batch size

            t0 = var(np.array([[0.0]]))#t=0
            u0 = self.calc(t0)#u(0)
            dudt0 = autograd(self.calc(t0))[t0]#u'(0)
            boundary_loss = (u0 - var(np.array([[1.0]])))**2 +dudt0**2

            total_loss = phyloss + var(np.array([[500.0]])) * boundary_loss#give more weight to boundary loss

            grads = autograd(total_loss)
            for layer in self.layers:
                layer.adam(grads,lr,0.9,0.999,1e-8)

            if e % 1000 == 0:
                print(f"epoch {e}  loss: {total_loss.val[0][0]:.6f}")


#ode is d2u/dt2=-u and boundary point is u(0)=1 and u'(0)=0
np.random.seed(50)


pinn1 = nn([1, 64,64,64, 1])
t_collocation = np.linspace(0,3*np.pi, 300)
pinn1.learn_pinn(epochs=30000, lr=0.001)
t_test=np.linspace(0,3*np.pi, 300)
u_analytical=np.cos(t_test)
u_pinn=[pinn1.calc(var(np.array([[t]]))).val[0][0] for t in t_test]

plt.plot(t_test, u_analytical, label='Analytical: cos(t)', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.show()
