if __name__=="_main_":
    #多层感知机的实现
    import torch
    from torch import nn
    import numpy as np
    from d2l import torch as d2l
    import matplotlib.pyplot as plt
    batch_size=256
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
    num_inputs,num_outputs,num_hiddens=784,10,256
    W1=nn.Parameter(torch.randn(num_inputs,num_hiddens,requires_grad=True)*0.01)
    b1=nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
    W2=nn.Parameter(torch.randn(num_hiddens,num_outputs,requires_grad=True)*0.01)
    b2=nn.Parameter(torch.zeros(num_outputs,requires_grad=True))
    params=[W1,b1,W2,b2]
    def relu(X):
        a=torch.zeros_like(X)
        return torch.max(a,X)
    def net(X):
        X=X.reshape((-1,num_inputs))
        H=relu(X@W1+b1)
        return(H@W2+b2)
    loss=nn.CrossEntropyLoss(reduction='none')
    num_epochs,lr=10,0.1
    updater=torch.optim.SGD(params,lr=lr)
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)
    d2l.predict_ch3(net,test_iter)
    plt.show()