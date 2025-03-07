import numpy as np
from numpy import ndarray
from typing import Callable

class Layer:
    def forward(self, X: ndarray) -> ndarray:
        raise NotImplementedError("Forward method not implemented")
    
    def backward(self, output_grad: ndarray, lr: float) -> ndarray:
        raise NotImplementedError("Backward method not implemented")

class WeightInitializer:
    @staticmethod
    def xavier_init(input_dim: int, output_dim: int) -> ndarray:
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

    @staticmethod
    def he_init(input_dim: int, output_dim: int) -> ndarray:
        stddev = np.sqrt(2 / input_dim)
        return np.random.normal(0, stddev, size=(input_dim, output_dim))

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, *, initializer: Callable[[int,int], ndarray] = WeightInitializer.he_init):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.weights: ndarray = initializer(input_size,output_size)
        self.bias: ndarray = np.zeros((1,output_size))
    
    def forward(self, X, train=True):
        if train:
            self.inputs = X
        return X.dot(self.weights) + self.bias
    
    def backward(self, output_grad, lr):
        grad = self.inputs.T.dot(output_grad)
        prev_grad = output_grad.dot(self.weights.T)
        self.weights -= lr * grad
        self.bias -= lr * np.sum(output_grad, axis=0, keepdims=True)
        return prev_grad

class Activation(Layer):
    def __init__(self, func, grad):
        self.func:  Callable[[ndarray], ndarray] = func
        self.grad:  Callable[[ndarray], ndarray] = grad
    def forward(self, X, train=True):
        if train:
            self.inputs = X
        return self.func(X)
    
    def backward(self, output_grad, lr):
        return output_grad * self.grad(self.inputs)

class ReLU(Activation):
    def __init__(self):
        func = lambda x: np.maximum(0, x)  # noqa: E731
        grad = lambda x: np.where(x < 0, 0, 1) # noqa: E731
        super().__init__(func, grad )


class LeakyReLU(Activation):
    def __init__(self):
        func = lambda x: np.where(x < 0,0.1 * x, x) # noqa: E731
        grad = lambda x: np.where(x < 0,0.1, 1) # noqa: E731
        super().__init__(func, grad)

class Tanh(Activation):
    def __init__(self):
        func = lambda x: (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) # noqa: E731
        grad = lambda x: 1 - x**2 # noqa: E731
        super().__init__(func, grad)
    def forward(self, X, train=True):
        if train:
            self.inputs = X
        self.output = self.func(X)
        return self.output
    def backward(self, output_grad, lr):
        return output_grad * self.grad(self.output)
        
class Sigmoid(Activation):
    def __init__(self):
        func = lambda x: 1/(1 + np.exp(-x)) # noqa: E731
        grad = lambda x: x * (1 - x) # noqa: E731
        super().__init__(func, grad)
    def forward(self, X, train=True):
        if train:
            self.inputs = X
        self.output = self.func(X)
        return self.output
    def backward(self, output_grad, lr):
        return output_grad * self.grad(self.output)

class Loss:
    def base(self, y_true: ndarray, y_pred: ndarray) -> float:
        raise NotImplementedError("Base method not implemented for loss function")
    
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        raise NotImplementedError("Grad method not implemented for loss function")

    
class MSE(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> float:
        return np.mean((y_true - y_pred)**2)
    
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return (y_pred - y_true) / len(y_true)
    
    
class BCE(Loss):
    def base(self, y_true: ndarray, y_pred: ndarray) -> float:
        return np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def grad(self, y_true: ndarray, y_pred: ndarray) -> ndarray:
        return 2 * (y_pred - y_true) / len(y_true)

class Sequential:
    def __init__(self, layers: list[Layer],*,loss: Loss = None, lr: float = 0.01, epochs: int = 1):
        self.layers: list[Layer] = layers
        self.lr: float = lr
        self.epochs: int = epochs
        self.loss = loss or MSE()
    
    def forward(self, X: ndarray, train=True):
        inputs = X
        for layer in self.layers:
            inputs = layer.forward(inputs, train)
        return inputs

    def backward(self, output_grad: ndarray):
        grad = output_grad
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, self.lr)
        
        
    def __call__(self, X: ndarray, y: ndarray, batch_size: int = 0):
        targets = y
        n_samples = X.shape[0]
        batch_size = batch_size or X.shape[0]
        
        if len(targets.shape) == 1:
            targets = y[:,np.newaxis]
        
        for e in range(self.epochs):
            batch_idx: ndarray = np.random.permutation(n_samples)
            batch_epochs = (n_samples // batch_size) + ( 1 if (n_samples % self.batch_size != 0) else 0 )
            
            for i in range(batch_epochs):
                start_batch_idx, end_batch_idx = i * self.batch_size, (i+1) * self.batch_size 
                batch_inputs = X[batch_idx[start_batch_idx:end_batch_idx], :]
                batch_targets = targets[batch_idx[start_batch_idx:end_batch_idx],:]
                batch_outputs = self.forward(batch_inputs)
                output_grad = self.loss.grad(batch_targets, batch_outputs)
                self.backward(output_grad)
        
    def predict(self, X: ndarray):
        return self.forward(X, train=False)