import torch

x = torch.tensor(2.0)
w = torch.tensor(2.0, requires_grad= True)

y = x * w

loss = (10 - y)** 2

loss.backward()

print("Prediction: ", y.item())
print("Loss: ", loss.item())
print("Gradient: ", w.grad.item())
