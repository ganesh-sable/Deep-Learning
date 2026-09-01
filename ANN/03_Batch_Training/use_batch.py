import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0]
])


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1,1)

    def forward(self, x):
        x = self.layer(x)
        return x

model =MyModel()

loss_fea = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

dataset = TensorDataset(X, y)

dataloder = DataLoader(
    dataset,
    batch_size= 2,
    shuffle= True
)

for epoch in range(1000):
    for batch_X , batch_y in dataloder:
        prediction = model(batch_X)

        loss = loss_fea(prediction, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    if epoch % 100 ==0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(model.layer.weight.item())

print(model.layer.bias.item())

test = torch.tensor([[6.0]])

output = model(test)

print("\nPrediction for x=6:")
print(output.item())