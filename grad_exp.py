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

    def __rsub__(self, other):
        return fnsub(var(other), self)
    def __rmul__(self, other):
        return fnmul(var(other), self)

def fnsum(var1, var2):
    x = lambda g: g
    return var(var1.val + var2.val, [var1, var2], [x, x])

def fnmul(var1, var2):
    val1 = var1.val.copy()
    val2 = var2.val.copy()
    x1 = lambda g: var(g.val * val2) if isinstance(g, var) else var(g * val2)
    x2 = lambda g: var(g.val * val1) if isinstance(g, var) else var(g * val1)
    return var(var1.val * var2.val, [var1, var2], [x1, x2])
def fnmatmul(var1, var2):
    val1 = var1.val.copy()
    val2 = var2.val.copy()
    
    def get_val(g):
        g_val = g.val if isinstance(g, var) else np.array(g)  # handles int/float too
        if g_val.ndim == 0:
            g_val = g_val.reshape(1, 1)
        return g_val
    
    def x1(g):
        return var(get_val(g) @ val2.T)
    
    def x2(g):
        return var(val1.T @ get_val(g))
    
    return var(var1.val @ var2.val, [var1, var2], [x1, x2])
def fnsigmoid(v):
    sig = 1 / (1 + np.exp(-v.val))
    sig_var = var(sig)
    x = lambda g: g * sig_var * (var(1) - sig_var)
    return var(sig, [v], [x])

def fnsub(var1, var2):
    x1= lambda g: g 
    x2= lambda g: g * var(-1)
    return var(var1.val - var2.val,[var1, var2], [x1, x2])

def fnpow(v, power):
    val_og = v.val.copy()
    x = lambda g: g*var(power* (val_og **(power - 1)))
    return var(v.val**power, [v], [x])
def fnsum_all(v):
    x=lambda g: g*var(np.ones_like(v.val)) 
    return var(np.sum(v.val), [v], [x])


def autograd(t, wrt=None):
    #Step 1:topological sort
    topo = []
    visited = set()
    def build_topo(node):
        if id(node) not in visited:
            visited.add(id(node))
            for parent in node.parents:
                build_topo(parent)
            topo.append(node)
    build_topo(t)

    #Step 2:accumulate gradients in reverse topological order
    grads={}
    grads[t]=var(np.ones_like(t.val))
    for node in reversed(topo):
        if node not in grads:
            continue
        g=grads[node]
        for parent, chainrule in zip(node.parents, node.chainrules):
            contrib=chainrule(g)
            if parent in grads:
                grads[parent] = var(grads[parent].val + contrib.val)
            else:
                grads[parent] = contrib
    return grads

# a = var(5)
# b=var(7)
# f=(a+b)*b
# dtdt=var(1)
# dv=autograd(f)
# dvda=dv[a]
# print(f.val,dvda.val)
