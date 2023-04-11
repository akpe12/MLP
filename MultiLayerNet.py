import numpy as np
import pickle
from collections import OrderedDict
from nn.functions import (
    Affine,
    Relu,
    SoftmaxWithLoss,
    he_normal,
    softmax,
)

class MultiLayerNet:
    def __init__(self, input_size: int, hidden_layer_num=1, hidden_size_list=[50], *args, output_size: int, weight_init_std='he') -> None:
        self.params = {}
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_size_list = hidden_size_list

        self.init_weight(weight_init_std)

        self.layers = OrderedDict()
        for i in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            self.layers['Activation' + str(i)] = Relu()
    
        i = self.hidden_layer_num + 1
        self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
        self.last_layer = SoftmaxWithLoss()

    def init_weight(self, weight_init_std: str) -> None:
        model_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for i in range(1, len(model_size_list)):
            scaling = weight_init_std
            if str(scaling).lower() == 'he':
                scaling = he_normal(model_size_list[i-1])
            self.params['W' + str(i)] = scaling * np.random.randn(model_size_list[i-1], model_size_list[i])
            self.params['b' + str(i)] = np.zeros(model_size_list[i])

    def __call__(self, x: np.array, y: np.array) -> float:
        loss = self.loss(x, y)
        
        return loss

    def predict(self, x: np.array) -> np.array:
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def inference(self, x: np.array, t: np.array, size: int) -> np.array:
        index = np.random.randint(0, len(x), size=size)
        x = self.predict(x[index])
        y = softmax(x)
        t = t[index].argmax(-1)

        return (y.argmax(-1), t)
    
    def loss(self, x: np.array, t: np.array) -> None:
        y = self.predict(x)

        return self.last_layer.forward(y, t)
    
    def accuracy(self, x: np.array, t: np.array) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def gradient(self) -> dict:
        dout = self.last_layer.backward()

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            grads['W' + str(i)] = self.layers['Affine' + str(i)].dW
            grads['b' + str(i)] = self.layers['Affine' + str(i)].db

        return grads
    
    def parameters(self) -> dict:
        return self.params
    
    def load_model(self, file_name: str) -> bool:
        with open(file_name, 'rb') as fr:
            params = pickle.load(fr)
        
        for i in range(1, self.hidden_layer_num + 2):
            self.params['W' + str(i)] = params['W' + str(i)]
            self.params['b' + str(i)] = params['b' + str(i)]
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])

        return True