import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size = 0.5,
    random_state = 42
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
    dtype = torch.float32
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
    batch_size = 32,
    shuffle= False
)

class CancerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(30, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):

        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)

        return x

model = CancerModel()

loss_fun = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)


for epoch in range(50):
    model.train()
    total_train_loss = 0

    for batch_x, batch_y in train_loader:
        y_pred = model(batch_x)

        trianLoss = loss_fun(y_pred, batch_y)

        optimizer.zero_grad()

        trianLoss.backward()

        optimizer.step()

        total_train_loss += trianLoss.item()

    average_train_loss = total_train_loss / len(train_loader)


    model.eval()
    correct = 0
    total = 0
    total_val_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)

            valLoss = loss_fun(y_pred, batch_y)

            total_val_loss += valLoss.item()

            probability = torch.sigmoid(y_pred)

            prediction = (probability >= 0.5).float()

            correct += ((prediction == batch_y).sum().item())

            total += batch_y.size(0)


    average_val_loss = total_val_loss / len(val_loader)

    validation_accuracy = correct/total 

    print(
        f"Epoch: {epoch}, "
        f"Train Loss: {average_train_loss:.4f}, "
        f"Val Loss: {average_val_loss:.4f}, "
        f"Val Accuracy: {validation_accuracy:.4f}"
    )


model.eval()

correct = 0
total = 0
total_test_loss = 0
actual_pre = []
all_prediction = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_pred = model(batch_x)

        testLoss = loss_fun(y_pred, batch_y)

        total_test_loss += testLoss.item()

        probability = torch.sigmoid(y_pred)

        prediction = (probability>=0.5).float()

        correct += ((prediction == batch_y).sum().item())

        total += batch_y.size(0)

        actual_pre.extend(batch_y.numpy().flatten())
        all_prediction.extend(prediction.numpy().flatten())

average_test_loss = total_test_loss / len(test_loader)

print("\nConfusion Matix: ")
print(confusion_matrix(actual_pre, all_prediction))

print("\nPrecision: ", precision_score(actual_pre, all_prediction))

print("\nRecall: ", recall_score(actual_pre, all_prediction))

print("\nF1 Score: ", f1_score(actual_pre, all_prediction))

test_accuracy = correct/ total

print("\nFinal Test Loss:", average_test_loss)

print("Final Test Accuracy:", test_accuracy)


new_data = X_test[0]

new_data = new_data.reshape(1, -1)
model.eval()

with torch.no_grad():
    prediction = model(new_data)

    probability = torch.sigmoid(prediction)

    pre = (probability>=0.5).float()

print("\nProbability: ", probability.item())
print("Prediction: ", pre.item())

print("Actual Value: ", y_test[0].item())


