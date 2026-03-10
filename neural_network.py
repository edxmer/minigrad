import random
from engine import Value

class Neuron:
    def __init__(self, input_size, normalise):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1, 1))
        self.normalise = normalise
    
    def __call__(self, inputs):
        pre = sum([i*w for (i, w) in zip(inputs, self.weights)], self.bias)
        act = self.normalise(pre)
        return act

    def arguments(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, prev_layer_size, layer_size, normalise):
        self.neurons = [Neuron(prev_layer_size, normalise) for _ in range(layer_size)]

    def __call__(self, prev_layer):
        activations = [n(prev_layer) for n in self.neurons]
        return activations
    
    def arguments(self):
        return [value for neuron in self.neurons for value in neuron.arguments()]

class MLP:
    def __init__(self, input_size, layer_sizes, normalise=Value.tanh, output_normalise=Value.tanh):
        combined = [input_size] + layer_sizes
        self.layers = [Layer(i_size, l_size, normalise) for (i_size, l_size) in zip(combined, layer_sizes[:-1])]
        self.layers.append(Layer(combined[-2], combined[-1], output_normalise))
    
    def __call__(self, input_layer):
        prev_out = [Value(x) for x in input_layer]
        for layer in self.layers:
            prev_out = layer(prev_out)
        return prev_out

    def arguments(self):
        return [value for layer in self.layers for value in layer.arguments()]