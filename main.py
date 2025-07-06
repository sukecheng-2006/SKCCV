# import torch
# from time import time
# # x=torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
# # y=2*x
# # z=y.view(2,2)
# # v=torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
# # l=torch.sum(z*v)
# # #print(l)
# # z.backward(v)
# # print(x.data.requires_grad)
# # x=torch.ones(1,requires_grad=True)
# # # print(x.data)
# # y=2*x
# # y.backward()
# # print(x.grad)
# a=torch.ones(1000)
# b=torch.ones(1000)
# start=time()
# # c=torch.zeros(1000)
# # for i in range(1000):
# #     c[i]=a[i]+b[i]
# d=a+b
# print(time()-start)
%matplotlib inline