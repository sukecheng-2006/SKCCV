"""从零实现 Fashion-MNIST 的单隐藏层多层感知机。"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


def relu(X: torch.Tensor) -> torch.Tensor:
    """对输入张量逐元素应用 ReLU 激活函数。"""
    zeros = torch.zeros_like(X)
    return torch.max(zeros, X)


def build_network(num_inputs: int, num_hiddens: int, num_outputs: int):
    """创建从零开始实现的网络参数和前向传播函数。"""
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    def net(X: torch.Tensor) -> torch.Tensor:
        X = X.reshape((-1, num_inputs))
        H = relu(X @ W1 + b1)
        return H @ W2 + b2

    return params, net


def train_from_scratch(num_epochs: int = 10, batch_size: int = 256, lr: float = 0.1) -> None:
    """使用手写前向传播函数训练并可视化 MLP。"""
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    params, net = build_network(num_inputs=784, num_hiddens=256, num_outputs=10)

    loss = nn.CrossEntropyLoss(reduction="none")
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)
    plt.show()


if __name__ == "__main__":
    train_from_scratch()
