import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = fetch_california_housing()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(
    X_train,
    dtype= torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype = torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype= torch.float32
).reshape(-1, 1)

y_test = torch.tensor(
    y_test,
    dtype= torch.float32
).reshape(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size= 32,
    shuffle= True
)

test_loader = DataLoader(
    test_dataset,
    batch_size= 32,
    shuffle= False
)

class HousingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(8, 64)
        self.dropout = nn.Dropout(p = 0.2)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.output(x)

        return x

model = HousingModel()

loss_fun = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

for epoch in range(50):
    total_loss = 0
    model.train()

    for X_batch, y_batch in train_loader:

        y_pred = model(X_batch)

        loss = loss_fun(y_pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

model.eval()
total_test_loss = 0

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        y_pred = model(X_batch)

        loss = loss_fun(y_pred, y_batch)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_loader)
print("\ntest loss: ", avg_test_loss)


new_data = X_test[0]

new_data = new_data.reshape(1, -1)


model.eval()

with torch.no_grad():

    prediction = model(new_data)

print(
    "Predicted Value:",
    prediction.item()
)

print(
    "Actual Value:",
    y_test[0].item()
)
