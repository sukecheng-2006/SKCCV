"""使用 PyTorch 模块化接口训练一个简单的 Fashion-MNIST 多层感知机。"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


def init_weights(module: nn.Module) -> None:
    """为全连接层初始化较小的权重以稳定训练过程。"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.01)


def train_mlp(num_epochs: int = 10, batch_size: int = 256, lr: float = 0.1) -> None:
    """基于 Sequential 接口训练单隐藏层 MLP 并展示训练曲线。"""
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()


if __name__ == "__main__":
    train_mlp()
