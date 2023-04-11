import numpy as np

class SGD: # overshooting 문제 발생 가능
    def __init__(self, params: dict, lr=1e-1, momentum=0, nesterov=False) -> None:
        self.params = params
        self.lr = lr
        self.velocity = None
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self, grads: dict) -> None:
        if self.velocity == None:
            self.velocity = {}
            for k, v in self.params.items():
                self.velocity[k] = np.zeros_like(v)

        for i in range(1, len(grads.keys()) // 2 + 1):
            # if self.nesterov:
            #     pass
            # else:
            self.velocity['W' + str(i)] = (self.momentum * self.velocity['W' + str(i)]) + (self.lr * grads['W' + str(i)])
            self.velocity['b' + str(i)] = (self.momentum * self.velocity['b' + str(i)]) + (self.lr * grads['b' + str(i)])
            self.params['W' + str(i)] -= self.velocity['W' + str(i)]
            self.params['b' + str(i)] -= self.velocity['b' + str(i)]

class Adagrad: # 학습 조기 중단될 수 있는 문제 발생 가능 -> 곡면 변화량을 전체 경로의 기울기 벡터의 크기로 계산
    def __init__(self, params: dict, lr=1e-1) -> None:
        self.params = params
        self.lr = lr
        self.h = None

    def step(self, grads: dict) -> None:
        eps = 1e-8
        if self.h == None:
            self.h = {}
            for k, v in self.params.items():
                self.h[k] = np.zeros_like(v)
        
        # h는 계속 누적되어 root h는 0이 될 가능성 존재. 그렇다고 root를 제거하면 성능이 나빠진다고 함.
        for i in range(1, len(grads.keys()) // 2 + 1):
            self.h['W' + str(i)] += grads['W' + str(i)] * grads['W' + str(i)]
            self.h['b' + str(i)] += grads['b' + str(i)] * grads['b' + str(i)]
            self.params['W' + str(i)] -= self.lr * grads['W' + str(i)] / (np.sqrt(self.h['W' + str(i)]) + eps)
            self.params['b' + str(i)] -= self.lr * grads['b' + str(i)] / (np.sqrt(self.h['b' + str(i)]) + eps)

# 초기 경로 편향 문제 발생, 초반에 step이 클 가능성
class RMSprop: # An extension of Adagrad, the sum of gradients is recursively defined as a decaying average of all past squared gradients -> rather averaging than accumulating gradients
    def __init__(self, params: dict, lr=1e-1, alpha=0.9) -> None:
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.da = None

    def step(self, grads: dict) -> None:
        eps = 1e-8
        if self.da == None:
            self.da = {}
            for k, v in self.params.items():
                self.da[k] = np.zeros_like(v)

        for i in range(1, len(grads.keys()) // 2 + 1):
            self.da['W' + str(i)] = self.alpha * self.da['W' + str(i)] + ((1 - self.alpha) * (grads['W' + str(i)] * grads['W' + str(i)]))
            self.da['b' + str(i)] = self.alpha * self.da['b' + str(i)] + ((1 - self.alpha) * (grads['b' + str(i)] * grads['b' + str(i)]))

            self.params['W' + str(i)] -= self.lr * grads['W' + str(i)] / (np.sqrt(self.da['W' + str(i)]) + eps)
            self.params['b' + str(i)] -= self.lr * grads['b' + str(i)] / (np.sqrt(self.da['b' + str(i)]) + eps)

class Adadelta: # 파라미터 업데이트 값의 지수가중이동평균 고려하여 lr 조절
    def __init__(self, params: dict, lr=1.0, alpha=0.9) -> None:
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.da_for_params = None
        self.da_for_grads = None

    def step(self, grads: dict) -> None:
        eps = 1e-8
        if self.da_for_params == None and self.da_for_grads == None:
            self.da_for_params = {}
            self.da_for_grads = {}
            for k, v in self.params.items():
                self.da_for_params[k] = np.zeros_like(v)
                self.da_for_grads[k] = np.zeros_like(v)

        for i in range(1, len(grads.keys()) // 2 + 1):
            self.da_for_params['W' + str(i)] = self.alpha * self.da_for_params['W' + str(i)] + ((1 - self.alpha) * (self.params['W' + str(i)] * self.params['W' + str(i)]))
            self.da_for_params['b' + str(i)] = self.alpha * self.da_for_params['b' + str(i)] + ((1 - self.alpha) * (self.params['b' + str(i)] * self.params['b' + str(i)]))
            self.da_for_grads['W' + str(i)] = self.alpha * self.da_for_grads['W' + str(i)] + ((1 - self.alpha) * (grads['W' + str(i)] * grads['W' + str(i)]))
            self.da_for_grads['b' + str(i)] = self.alpha * self.da_for_grads['b' + str(i)] + ((1 - self.alpha) * (grads['b' + str(i)] * grads['b' + str(i)]))

            self.params['W' + str(i)] -= self.lr * (grads['W' + str(i)] * (np.sqrt(self.da_for_params['W' + str(i)])) + eps) / (np.sqrt(self.da_for_grads['W' + str(i)]) + eps)
            self.params['b' + str(i)] -= self.lr * (grads['b' + str(i)] * (np.sqrt(self.da_for_params['b' + str(i)])) + eps) / (np.sqrt(self.da_for_grads['b' + str(i)]) + eps)

# 수정 필요
class Adam: # momentum + rmsprop, 초기값인 0.1과 0.01이 상쇄되며 편향 제거
    def __init__(self, params: dict, lr=1e-1, betas=(0.9, 0.999)) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.velocity = None
        self.da = None

    def step(self, grads: dict) -> None:
        if self.velocity == None and self.da == None:
            self.velocity = {}
            self.da = {}
            for k, v in self.params.items():
                self.velocity[k] = np.zeros_like(v)
                self.da[k] = np.zeros_like(v)
        
        for i in range(1, len(grads.keys()) // 2 + 1):
            eps = 1e-8
            self.velocity['W' + str(i)] = (self.beta1 * self.velocity['W' + str(i)]) + (grads['W' + str(i)] * (1 - self.beta1))
            self.velocity['b' + str(i)] = (self.beta1 * self.velocity['b' + str(i)]) + (grads['b' + str(i)] * (1 - self.beta1))
            self.da['W' + str(i)] = self.beta2 * self.da['W' + str(i)] + ((1 - self.beta2) * (grads['W' + str(i)] * grads['W' + str(i)]))
            self.da['b' + str(i)] = self.beta2 * self.da['b' + str(i)] + ((1 - self.beta2) * (grads['b' + str(i)] * grads['b' + str(i)]))

            # 첫 번째 스텝에서 기울기값이 너무 작으면 첫 스텝을 너무 크게 가져가서 편향을 가진 학습이 진행될 수 있다. 그것을 방지하기 위한 개선식 구현
            # 그러나, 두 번째 스텝부터 1 - beta1 값이 곱해지지 않은 기울기값이 그대로 velocity에 들어가고,
            # 1 - beta2 값이 곱해지지 않은 기울기의 제곱값이 da에 들어가서 값이 너무 커져서 학습이 제대로 진행되지 않을 가능성 존재하게 됨.
            # self.velocity['W' + str(i)] = self.velocity['W' + str(i)] / (1 - self.beta1)
            # self.velocity['b' + str(i)] = self.velocity['b' + str(i)] / (1 - self.beta1)
            # self.da['W' + str(i)] = self.da['W' + str(i)] / (1 - self.beta2)
            # self.da['b' + str(i)] = self.da['b' + str(i)] / (1 - self.beta2)

            self.params['W' + str(i)] -= self.lr * (self.velocity['W' + str(i)] / (np.sqrt(self.da['W' + str(i)]) + eps))
            self.params['b' + str(i)] -= self.lr * (self.velocity['b' + str(i)] / (np.sqrt(self.da['b' + str(i)]) + eps))