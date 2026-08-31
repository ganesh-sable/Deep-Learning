import torch 
import torch.nn as nn

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0]
])

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer(x)

        return x


model = MyModel()

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

for epoch in range(1000):

    prediction = model(X)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    print(
            f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}"
        )

print("\nWeight:")
print(model.layer.weight)

print("\nBias:")
print(model.layer.bias)


test = torch.tensor([[6.0]])

prediction = model(test)

print("\nPrediction for x=6:")
print(prediction.item())

