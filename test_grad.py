import torch
x = torch.tensor([1.0, 2.0, 4.0, 7.0])
with torch.no_grad():
    dx = torch.gradient(x)[0]
    print(dx)
