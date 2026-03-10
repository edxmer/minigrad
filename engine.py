import math

class Value:
    def __init__(self, data, children=()):
        self.data = float(data)
        self.children = set(children)
        self.grad = 0.0
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            # Local derivative: 1.0
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            # Local derivative: f(a)=a*b -> f'(a)=b
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        other = Value(other)

        out = Value(self.data ** other.data, (self, other))
        def _backward():
            # Local derivative respect to self: f(s)=s^o -> f'(s)=o*s^(o-1)
            self.grad += other.data * self.data ** (other.data-1.0) * out.grad
        out._backward = _backward
        return out

    def __rpow__(self, other):
        return self ** other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other): # other - self
        return other + -self

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return self / other

    def relu(self):
        out = Value(self.data if 0 < self.data else 0, (self, ))
        def _backward():
            self.grad += float(0 < self.data) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ))
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        #print("tanh", self.data)
        out = Value(math.tanh(self.data), (self, ))
        def _backward():
            self.grad += (1 - out.data*out.data) * out.grad
        out._backward = _backward
        return out

    def backward(self, zero_grad=True):
        # **** Topological sort
        deg: list[list[Value]] = [[self]]
        while deg[-1] != []:
            deg.append([x for xs in deg[-1] for x in xs.children])

        seen = set()
        sorted_list:list[Value]=[]

        for xs in reversed(deg):
            for x in xs:
                if not x in seen:
                    seen.add(x)
                    sorted_list.append(x)
        
        # **** 
        if zero_grad:
            for x in reversed(sorted_list):
                x.grad = 0

        self.grad = 1.0
        for x in reversed(sorted_list):
            x._backward()

