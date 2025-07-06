import torch
x=torch.ones(2,2,requires_grad=True)
out=x.sum()
out.backward()
