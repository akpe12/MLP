import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)

    return np.exp(x) / np.sum(np.exp(x))

# xaviers for activation functions with linear(e.g. sigmoid, tanh)
def xavier_normal(in_size, out_size):
    return np.sqrt(2. / (in_size + out_size))


def xavier_uniform(in_size, out_size):
    return np.sqrt(6. / (in_size + out_size))


def he_normal(in_size):
    return np.sqrt(2. / in_size)


def he_uniform(in_size):
    return np.sqrt(6. / in_size)

def cross_entropy_loss(pred: np.array, label: np.array):
    e = 1e-7
    batch_size = pred.shape[0]
    
    return np.sum(label * np.log(pred + e)) / batch_size

class Sigmoid:
    def __init__(self) -> None:
        self.out = None
    
    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * ((1. - self.out) * self.out)

        return dx
    
class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Affine:
    def __init__(self, W, b) -> None:
        self.W = W
        self.b = b
        self.x = None
        self.original_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) # delta
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0) # 행벡터 데이터

        dx = dx.reshape(*self.original_shape)
        
        return dx
    
class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None   # softmax의 out
        self.t = None   # 원핫인코딩

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y, self.t)

        return self.loss
    
    def backward(self):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx