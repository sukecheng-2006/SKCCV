import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
net=nn.Sequential(nn.Flatten(),
                  nn.Linear(784,256),
                  nn.ReLU(),
                  nn.Linear(256,10)
                  )
def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)
loss=nn.CrossEntropyLoss(reduction='none')
lr=0.1
trainer=torch.optim.SGD(net.parameters(),lr=lr)
train_iter,test_iter=d2l.load_data_fashion_mnist(256)
d2l.train_ch3(net,train_iter,test_iter,loss,10,trainer)
plt.show()