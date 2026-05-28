import numpy as np  
class layer():
    
    def __init__(self,numnodesin,numnodesout):
        self.numnodesin=numnodesin
        self.numnodesout=numnodesout
        self.weights=2*np.random.random((numnodesout,numnodesin))-1
        self.biases=2*np.random.random((numnodesout,1))-1
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
    def calcoutputs(self,inputs):
        outs=(np.dot(self.weights,inputs))
        outs+=self.biases
        x=self.Activationfunc(outs)
        self.layeratt["acts"]=x
        self.layeratt["inputs"]=inputs
        return x
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
    def calc(self,inputs):
        for x in self.layers:
            inputs=x.calcoutputs(inputs)#output becomes the next input
        return inputs
    def classify(self,inputs):#choose the final answer
        outs=self.calc(inputs)
        return np.argmax(outs)
    def cost(self,datapoint):
        out=self.calc(datapoint[0])
        costarr=((out-datapoint[1])**2)/2
        cost=np.sum(costarr)
        return cost
    def costtotal(self, data):
        totalcost = sum(self.cost(d) for d in data)
        return totalcost / len(data)
    def updateallgrads(self,datapoint):
        self.calc(datapoint[0])
        outputlayer=self.layers[-1]
        nodevalues=outputlayer.calcoutlayernodevalues(datapoint[1])#output layer need different calculation than hidden layers
        outputlayer.updategrads(nodevalues)
        for i in range(len(self.layers)-2,-1,-1):
            hiddenlayer=self.layers[i]
            nodevalues=hiddenlayer.hiddennodevalues(self.layers[i+1],nodevalues)
            hiddenlayer.updategrads(nodevalues)
    def applyallgrad(self,lr):
        for i in self.layers:
            i.applygrad(lr)

    def learn(self,trainbatch,lr):
        for layer in self.layers:
            layer.costgradientw = np.zeros_like(layer.costgradientw)
            layer.costgradientb = np.zeros_like(layer.costgradientb)
        for i in trainbatch:
            self.updateallgrads(i)
        self.applyallgrad(lr/len(trainbatch))#outside because apply once for the batch











#simple first test
# data=np.array([[[1,1],[0,1]],[[0,0],[0,1]],[[1,0],[1,0]],[[0,1],[1,0]]])    
# nn1=nn([2,1,2])
# print(nn1.cost([[1,1],[0,1]]))

np.random.seed(45)
#simple training with two category output
c1 = np.random.randn(50, 2) + np.array([1, 1])
y1 = np.tile([1, 0], (50, 1))

c2 = np.random.randn(50, 2) + np.array([4, 4])
y2 = np.tile([0, 1], (50, 1))

X = np.vstack([c1, c2])
y = np.vstack([y1, y2])

idx = np.random.permutation(100)
X, y = X[idx], y[idx]
#taing 80% data for training, 20% for testing
X_train, y_train = X[:80], y[:80]
X_test,  y_test  = X[80:], y[80:]

train_data = [[X_train[i].reshape(-1,1), y_train[i].reshape(-1,1)] for i in range(len(X_train))]
test_data  = [[X_test[i].reshape(-1,1),  y_test[i].reshape(-1,1)]  for i in range(len(X_test))]

nn1 = nn([2, 4, 2])
print("cost before training:", nn1.cost(train_data[0]))

epochs = 1000
for e in range(epochs):
    nn1.learn(train_data, lr=0.1)
    if e % 100 == 0:
        print(f"epoch {e}  cost: {nn1.costtotal(train_data):.4f}")

correct = 0
for d in test_data:
    pred   = nn1.classify(d[0])
    actual = np.argmax(d[1])
    if pred == actual:
        correct += 1

print(f"\nAccuracy: {correct}/{len(test_data)} = {correct/len(test_data)*100}%")