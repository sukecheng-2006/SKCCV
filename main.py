import torch
x=torch.tensor([1.0,2.0,3.0,4.0],requires_grad=True)
y=2*x
z=y.view(2,2)
v=torch.tensor([[1.0,0.1],[0.01,0.001]],dtype=torch.float)
l=torch.sum(z*v)
#print(l)
z.backward(v)
print(x.grad)

