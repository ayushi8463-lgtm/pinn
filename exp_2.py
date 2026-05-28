import numpy as np 
import matplotlib.pyplot as plt
from grad_exp import var, fnsum, fnmul, fnmatmul, fnsigmoid, autograd ,fnsum_all
class layer():
    
    def __init__(self,numnodesin,numnodesout):
        self.numnodesin=numnodesin
        self.numnodesout=numnodesout
        self.weights = var(2 * np.random.random((numnodesout, numnodesin)) - 1)
        self.biases = var(2 * np.random.random((numnodesout, 1)) - 1)
        self.costgradientw=np.zeros((numnodesout,numnodesin))
        self.costgradientb=np.zeros((numnodesout,1))
        self.layeratt={}#to store data so that the calculations dont need to be done multiple times
    def applygrad(self,lr):
        self.biases-=self.costgradientb*lr
        self.weights-=self.costgradientw*lr
    def Activationfunc(self,x,deriv=False):
        if deriv==True:
            return x*(1-x)    
        return 1/(1+np.exp(-x))
    def calcoutputs(self, inputs,is_output=False):
        if not isinstance(inputs, var):
            inputs = var(inputs)
        z = fnmatmul(self.weights, inputs) + self.biases
        if is_output:
            return z  
        return fnsigmoid(z)
    def calcoutlayernodevalues(self,expoutputs):#for output layer only
        dcda=self.layeratt["acts"]-expoutputs
        dadz=self.Activationfunc(self.layeratt["acts"], True)
        x=dadz*dcda
        self.layeratt["outnodevalues"]=x
        return x
    def updategrads(self,nodevalues): #updating gradients
        self.costgradientw+=np.dot(nodevalues,self.layeratt["inputs"].T)
        self.costgradientb+=nodevalues
    def hiddennodevalues(self,oldlayer,oldnodevalues):#for hidden layers
        weightedinderiv=np.dot(oldlayer.weights.T, oldnodevalues  )      
        newnodevalues=self.Activationfunc(self.layeratt["acts"],True)*weightedinderiv
        return newnodevalues


class nn():
    def __init__(self,layersizes):
        self.layersizes=layersizes
        self.layers=[]
        for i in range(len(layersizes)-1):
            self.layers.append(layer(self.layersizes[i],self.layersizes[i+1]))
    def calc(self, inputs):
        for i, x in enumerate(self.layers):
            is_output = (i == len(self.layers) - 1)
            inputs = x.calcoutputs(inputs, is_output=is_output)
        return inputs
    def classify(self,inputs):#choose the final answer
        outs=self.calc(inputs)
        return np.argmax(outs.val)
    def cost(self,datapoint):
        out=self.calc(datapoint[0])
        costarr=((out-datapoint[1])**2)*0.5
        cost=np.sum(costarr.val)
        return cost
    def costtotal(self, data):
        totalcost = sum(self.cost(d) for d in data)
        return totalcost / len(data)
    # def learn(self, trainbatch, lr):
    #     weight_grads = {l.weights: np.zeros_like(l.weights.val) for l in self.layers}
    #     bias_grads = {l.biases: np.zeros_like(l.biases.val) for l in self.layers}

    #     for datapoint in trainbatch:
    #         inputs = var(datapoint[0])
    #         expected= datapoint[1]
    #         out=self.calc(inputs)
    #         loss=fnsum_all((out - var(expected)) ** 2 * 0.5)
    #         grads=autograd(loss)

    #         for l in self.layers:
    #             if l.weights in grads:
    #                 weight_grads[l.weights] += grads[l.weights]
    #             if l.biases in grads:
    #                 bias_grads[l.biases] += grads[l.biases]

    #     for l in self.layers:
    #         l.weights.val-=lr / len(trainbatch) * weight_grads[l.weights]
    #         l.biases.val-=lr / len(trainbatch) * bias_grads[l.biases]
    def learn_pinn(self,epochs,lr):
        for e in range(epochs):
            phyloss=var(0.0)
            for t in t_collocation:
                t_var=var(np.array([[t]]))
                u= self.calc(t_var)#forward pass
                dudt=autograd(u,t_var)[t_var]
                residual=dudt + u
                phyloss=phyloss+fnsum_all(residual ** 2)
            #boundary condition u(0)=1
            t0=var(np.array([[0.0]]))
            u0 =self.calc(t0)
            boundary_loss=fnsum_all((u0-var(np.array([[1.0]])))**2)
            total_loss = phyloss + var(np.array(50.0)) * boundary_loss  # weight boundary more
            # update weights
            grads=autograd(total_loss)
            for layer in self.layers:
                if layer.weights in grads:
                    layer.weights.val -= lr * grads[layer.weights].val
                if layer.biases in grads:
                    layer.biases.val -= lr * grads[layer.biases].val
            
            if e % 1000 == 0:
                print(f"epoch {e}  loss: {total_loss.val:.4f}")


#ode is du/dt=-u and boundary point is u(0)=1
pinn1 = nn([1, 20, 20, 1])


t_collocation=np.linspace(0, 2, 100)#100 points from t=0 to 2
pinn1 = nn([1, 20, 20, 1])
pinn1.learn_pinn(epochs=10000, lr=0.001)
# compare against analytical solution u(t) = e^(-t)
t_test=np.linspace(0, 2, 200)
u_analytical=np.exp(-t_test)
u_pinn=[pinn1.calc(var(np.array([[t]]))).val[0][0] for t in t_test]

plt.plot(t_test, u_analytical, label='Analytical: e^(-t)', color='blue')
plt.plot(t_test, u_pinn, label='PINN', color='red', linestyle='--')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('PINN vs Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()

