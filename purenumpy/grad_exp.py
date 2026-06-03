import numpy as np 
class var:
    def __init__(self,val, parents=None,chainrules=None):
        self.val = np.array(val) if not isinstance(val, np.ndarray) else val
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

    def __rsub__(self, other):#supports subtraction even if other operand is not var object
        return fnsub(var(other), self)
    def __rmul__(self, other):#supports multiplication even if other operand is not var object
        return fnmul(var(other), self)
    def __truediv__(self, other):
        if not isinstance(other, var):
            other = var(other)
        return fndiv(self, other)

    def __rtruediv__(self, other):
        return fndiv(var(other), self)

def fnsum(var1, var2):
    def grad1(g):

        #this if statement is to fix an error encountered when adding bias velocity and momentum
        if g.val.shape != var1.val.shape:
            axis = (1,)#(n,batchsize) and (n,1) needed to be matched
            #sum the values to get batch size*g. 
            #This node doesnt have parents but we dont need it because we never differentiate twice wrt bw and bv
            return var(g.val.sum(axis=axis, keepdims=True))  
        return g
    def grad2(g):
        if g.val.shape != var2.val.shape:
            axis = (1,)
            return var(g.val.sum(axis=axis, keepdims=True))
        return g
    return var(var1.val + var2.val, [var1, var2], [grad1, grad2])

def fnsub(var1, var2):
    x1= lambda g: g 
    x2= lambda g: g * var(-1)
    return var(var1.val - var2.val,[var1, var2], [x1, x2])

def fnmul(var1, var2):
    x1 = lambda g: g*var2
    x2 = lambda g: g*var1
    return var(var1.val * var2.val, [var1, var2], [x1, x2])
def fnmatmul(var1, var2):
    x1 = lambda g: g @ var(var2.val.T)
    x2 = lambda g: var(var1.val.T) @ g
    return var(var1.val @ var2.val, [var1, var2], [x1, x2])
def fnexp(v):
    exp_val = np.exp(v.val)
    out=var(exp_val,[v],None)
    x = lambda g:g*out
    out.chainrules=[x]
    return out

def fndiv(var1, var2):
    x1 = lambda g: g /var2
    x2 = lambda g: g *var(-1)*var1/var2**2
    return var(var1.val / var2.val, [var1, var2], [x1, x2])
def fnsigmoid(v):
    sig=var(np.ones_like(v.val))/(var(np.ones_like(v.val))+fnexp(v*var(-1)))#1/(1+e^-v)
    return sig

def fnsum_all(v):
    x = lambda g: var(np.ones_like(v.val) * g.val)
    return var(np.array([[v.val.sum()]]), [v], [x])#summing the gradients for all points in a batch

def fnpow(var1, power):
    x = lambda g: g*var(power)* (var1 **(power - 1))
    return var(var1.val**power, [var1], [x])

def fntanh(v):
    return (fnexp(v) - fnexp(v * var(-1))) / (fnexp(v) + fnexp(v * var(-1)))



def autograd(t):
    #Step 1:topological sort
    topo= []
    visited= set()
    def build_topo(node):
        if id(node) not in visited:
            visited.add(id(node))
            for parent in node.parents:
                build_topo(parent)
            topo.append(node)
    build_topo(t)#topo will be list with initial elements as parent and children later

    #Step 2:accumulate gradients in reverse topological order
    grads={}
    grads[t]=var(np.ones_like(t.val))
    for node in reversed(topo):  #child first,parent later
        if node not in grads:
            continue
        g=grads[node]
        for parent, chainrule in zip(node.parents, node.chainrules):
            contrib=chainrule(g) 
            if parent in grads:
                grads[parent] =grads[parent]+ contrib
            else:
                grads[parent] = contrib
    return grads


# #small test run
# a = var(5)
# b=var(7)
# f=(a+b)*b
# dtdt=var(1)
# dv=autograd(f)
# dvda=dv[a]
# print(f.val,dvda.val)

# #supports higher order differentiation
# a = var(np.array(3.0))
# f = a ** 3
# dfda = autograd(f)[a]#3a^2 = 27
# d2fda2 = autograd(dfda)[a]#6a  = 18
# d3fda3 = autograd(d2fda2)[a]#6  = 6
# print(dfda.val, d2fda2.val, d3fda3.val) 