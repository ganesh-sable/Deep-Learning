import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0],
    [7.0],
    [8.0],
    [9.0],
    [10.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0],
    [15.0],
    [17.0],
    [19.0],
    [21.0]
])


X_train = X[:8]
y_train = y[:8]

X_test = X[8:]
y_test = y[8:]


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer(x)

        return x

model = MyModule()

loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.01
)

train_dataset = TensorDataset(X_train, y_train)

test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size = 2,
    shuffle= True
)

test_loader = DataLoader(
    test_dataset,
    batch_size= 2,
    shuffle= True
)

for epoch in range(1000):
    model.train()

    total_loss = 0

    for batch_X , batch_y in train_loader:

        y_pred = model(batch_X)

        loss = loss_fun(y_pred, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        average_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {average_loss:.4f}")


model.eval()

total_test_loss = 0
with torch.no_grad():

    for X_batch, y_batch in test_loader:

        y_pred = model(X_batch)

        loss = loss_fun(
            y_pred,
            y_batch
        )

        total_test_loss += loss.item()


average_test_loss = (
    total_test_loss / len(test_loader)
)

print("\nTest Loss:", average_test_loss)


new_data = torch.tensor([
    [11.0]
])

model.eval()

with torch.no_grad():

    prediction = model(new_data)


print("Prediction for 11:", prediction.item())