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

    def __rsub__(self, other):
        return self - other

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
        e = (self*2).exp()
        return (e-1)/(e+1)

    def backward(self, zero_grad=True):
        # Topological sort
        sorted_list=[]
        seen = set()
        def topological_sort(a:Value):
            if not a in seen:
                seen.add(a)
                for child in a.children:
                    topological_sort(child)
                sorted_list.append(a)
                if zero_grad:
                    a.grad = 0

        topological_sort(self)
        self.grad = 1.0
        for e in reversed(sorted_list):
            e._backward()
