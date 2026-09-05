import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()

X = data.data
y = data.target


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42
)

X_val, X_test, y_val, y_test = train_test_split(
    X, y,
    test_size= 0.5,
    random_state= 42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(
    X_train,
    dtype = torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype = torch.float32
).reshape(-1, 1)

X_val = torch.tensor(
    X_val,
    dtype = torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype = torch.float32
).reshape(-1, 1)

X_test = torch.tensor(
    X_test,
    dtype= torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype = torch.float32
).reshape(-1, 1)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


train_loader = DataLoader(
    train_dataset,
    batch_size= 32,
    shuffle= True
)

val_loader = DataLoader(
    val_dataset,
    batch_size= 32,
    shuffle= False
)

test_loader = DataLoader(
    test_dataset,
    batch_size= 32,
    shuffle= False
)

class HosuingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(8, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)

        return x

model = HosuingModel()

loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)


best_val_loss = float("inf")

patience = 5

counter = 0

best_model_state = None

for epoch in range(1000):
    model.train()
    total_train_loss = 0

    for batch_X, batch_y in train_loader:
        y_pred = model(batch_X)

        train_loss = loss_fun(y_pred, batch_y)

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

        total_train_loss += train_loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )
    

    model.eval()
    total_val_loss = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:
            y_pred = model(batch_X)

            val_loss = loss_fun(y_pred, batch_y)

            total_val_loss += val_loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )

    print(
        f"Epoch: {epoch}, "
        f"Train Loss: {average_train_loss:.4f}, "
        f"Validation Loss: {average_val_loss:.4f}"
    )

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        counter = 0

        best_model_state = model.state_dict()

    else:

        counter += 1
        if counter >= patience:
            break

model.load_state_dict(best_model_state)
print("\nBest Validation Loss:", best_val_loss)


model.eval()
total_test_loss = 0

with torch.no_grad():

    for batch_X, batch_y in test_loader:
        y_pred = model(batch_X)

        test_loss = loss_fun(y_pred, batch_y)

        total_test_loss += test_loss.item()

average_test_loss = (
        total_test_loss / len(test_loader)
    )

print("Final Test Loss: ", average_test_loss)


new_data = X_test[0]

new_data = new_data.reshape(1, -1)

model.eval()

with torch.no_grad():
    prediction = model(new_data)

print("Prediction: ",prediction.item())

print("Actual Value:", y_test[0].item())
        

