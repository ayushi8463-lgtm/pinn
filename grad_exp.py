import numpy as np 
class var:
    def __init__(self,val, parents=None,chainrules=None):
        self.val=val
        self.parents=parents if parents is not None else []
        self.chainrules=chainrules if chainrules is not None else []
    def __add__(self, other):
        if not isinstance(other, var):
            other=var(other)
        return fnsum(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, var):
            other=var(other)
        return fnmul(self, other)
    def __matmul__(self, other):
        if not isinstance(other, var):
            other = var(other)
        return fnmatmul(self, other)
    def __sub__(self, other):
        if not isinstance(other, var):
            other=var(other)
        return fnsub(self, other)
    def __pow__(self, power):
        return fnpow(self, power)#never need check here bcoz other parameter is a value
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return fnsub(var(other), self)
    def __rmul__(self, other):
        return fnmul(var(other), self)

def __pow__(self, power):
    return fnpow(self, power)
def fnmul(var1,var2):

    x1=lambda globaldvar: globaldvar*var2.val
    x2=lambda globaldvar:globaldvar*var1.val
    return var(var1.val*var2.val,[var1,var2],[x1,x2])
def fnsum(var1,var2):
    x=lambda globaldvar: globaldvar

    return var(var1.val+var2.val,[var1,var2],[x,x])
def fnmatmul(var1, var2):
    x1 = lambda g: g @ var2.val.T
    x2 = lambda g: var1.val.T @ g
    return var(var1.val @ var2.val, [var1, var2], [x1, x2])
def fnsigmoid(v):
    sig = 1 / (1 + np.exp(-v.val))
    x = lambda g: g * sig * (1 - sig)
    return var(sig, [v], [x])

def fnsub(var1, var2):
    x1= lambda g: g * 1
    x2= lambda g: g * -1
    return var(var1.val - var2.val,[var1, var2], [x1, x2])
def fnpow(v, power):
    x=lambda g: g * power * (v.val ** (power - 1))
    return var(v.val ** power, [v], [x])
def fnsum_all(v):
    x=lambda g: np.ones_like(v.val) * g
    return var(np.sum(v.val), [v], [x])

def autograd(t):
    grads=dict()
    def compute(node,globalgrad):
        for parent,chainrule in zip(node.parents,node.chainrules):
            newglobalgrad=chainrule(globalgrad)
            if parent in grads.keys():
                grads[parent]+=newglobalgrad
            else:
                grads[parent]=newglobalgrad
            compute(parent,newglobalgrad)
    dtdt=1
    compute(t,dtdt)
    return grads


a = var(5)
b=var(7)
f=(a+b)*b
dtdt=var(1)
dv=autograd(f)
dvda=dv[a]
print(f.val,dvda)
