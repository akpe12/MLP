import numpy as np
from sklearn import datasets

def Load_dataset(dataset_name="mnist_784"):
    X, y = datasets.fetch_openml(dataset_name, version=1, return_X_y=True)
    # 데이터 normalize 및 split
    X, y = X.to_numpy(), y.to_numpy()
    X = X.astype(np.float32)
    X /= 255.
    X -= X.mean(axis=0)

    return X, y

def Dataloader(x, y, batch_size=100, train=True):
    batch = []
    train_size = x.shape[0]
    if train:
        index = np.arange(train_size)
        np.random.shuffle(index)
        x = x[index]
        y = y[index]
    
    total_batch_num = train_size // batch_size

    for _ in range(total_batch_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]
        batch.append((x_batch, y_batch))

    return batch, total_batch_num

def grad_clip_norm(grads: dict, max_norm: int, error_if_nonfinite: bool = False) -> dict:
    max_norm = float(max_norm)
    for i in range(1, len(grads.keys()) // 2 + 1):
        norm_W = np.linalg.norm(grads['W' + str(i)])
        norm_b = np.linalg.norm(grads['b' + str(i)])

        if error_if_nonfinite and np.isnan(norm_W) or np.isnan(norm_b):
            raise RuntimeError("`grads` is non-finite, so it cannot be clipped")

        if np.isnan(norm_W):
            grads['W' + str(i)][np.isnan(grads['W' + str(i)])] = 1.0
            norm_W = np.linalg.norm(grads['W' + str(i)])
        if np.isnan(norm_b):
            grads['b' + str(i)][np.isnan(grads['b' + str(i)])] = 1.0
            norm_b = np.linalg.norm(grads['b' + str(i)])

        if norm_W > max_norm:
            grads['W' + str(i)] = (max_norm / norm_W) * grads['W' + str(i)]
        if norm_b > max_norm or np.isnan(norm_b):
            grads['b' + str(i)] = (max_norm / norm_b) * grads['b' + str(i)]

    return grads


class EarlyStopping:
    def __init__(self, patience=3, std="acc", improved_std=0.0001) -> None:
        self.patience = patience
        self.patience_curr = 0
        self.performance = 0
        self.std = std
        self.improved_std = improved_std

    def step(self, performance: float) -> bool:
        if self.std.lower()[:3] == "acc":
            if performance >= self.performance + self.improved_std:
                self.patience_curr = 0
                self.performance = performance
            else:
                self.patience_curr += 1

            if self.patience == self.patience_curr:
                return False
            else:
                return True
        elif self.std.lower() == "loss":
            if performance <= self.performance - self.improved_std:
                self.patience_curr = 0
                self.performance = performance
            else:
                self.patience_curr += 1

            if self.patience == self.patience_curr:
                return False
            else:
                return True