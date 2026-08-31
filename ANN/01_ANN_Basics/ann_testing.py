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

        self.layer = nn.Linear(1,1)

    def forward(self, x):
        x = self.layer(x)

        return x

model = MyModel()


loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

model.train()

for epoch in range(1000):

    y_pred = model(X)

    loss = loss_fun(y_pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


model.eval()


with torch.no_grad():

    predictions = model(X)
    test_loss = loss_fun(predictions, y)

print(test_loss.item())


test = torch.tensor([[6.0]])

prediction = model(test)

print(prediction.item())
