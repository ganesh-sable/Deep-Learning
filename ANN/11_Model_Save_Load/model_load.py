import torch
import torch.nn as nn

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)
print(set(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size= 0.2,
    random_state= 42,
    stratify = y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(
    X_train, 
    dtype = torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype = torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype = torch.long
)

y_test = torch.tensor(
    y_test,
    dtype = torch.long
)

class IrisModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.output(x)

        return x

model = IrisModel()

model.load_state_dict(torch.load("iris_model.pth"))


model.eval()

with torch.no_grad():

    output = model(X_test)

    prediction = torch.argmax(output, dim = 1)


correct = (prediction == y_test).sum().item()

total = y_test.size(0)

accuracy = correct / total

print("\nActual value: ", y_test)
print("\nPrediction: ", prediction)
print("\nAccuracy: ", accuracy)

