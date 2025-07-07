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
# import matplotlib_inline
# import torch
# from IPython import display
# from matplotlib import pyplot as plt
# import numpy as np
# import random
# num_inputs=2
# num_examples=1000
# true_w=[2,-3.4]
# true_b=4.2
# features=torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
# labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# labels+=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))

# print(labels)
