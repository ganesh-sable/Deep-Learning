import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, dataloader

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_iris()

X = data.data
y = data.target

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size= 0.1,
    random_state= 42,
    stratify= y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size= 1/9,
    random_state= 42,
    stratify= y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype = torch.float32)
X_val = torch.tensor(X_val, dtype = torch.float32)
X_test = torch.tensor(X_test, dtype = torch.float32)

y_train = torch.tensor(y_train, dtype = torch.long)
y_val = torch.tensor(y_val, dtype = torch.long)
y_test = torch.tensor(y_test, dtype = torch.long)

class CancerModel(nn.Module):

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

parameters = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "optimizers": ["Adam", "SGD"],
    "epochs": [100, 300, 500, 1000]
}

results = {}

for lr in parameters["learning_rate"]:

    for optimizer_name in parameters["optimizers"]:

        for epochs in parameters["epochs"]:

            model = CancerModel()

            criterion = nn.CrossEntropyLoss()

            if optimizer_name == "Adam":

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr
                )

            elif optimizer_name == "SGD":

                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr
                )

            # Training
            for epoch in range(epochs):

                model.train()

                output = model(X_train)

                loss = criterion(
                    output,
                    y_train
                )

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            # Validation
            model.eval()

            with torch.no_grad():

                output = model(X_val)

                prediction = torch.argmax(
                    output,
                    dim=1
                )

                accuracy = (
                    prediction == y_val
                ).float().mean().item()

            results[
                (lr, optimizer_name, epochs)
            ] = accuracy

            print(
                f"LR: {lr} | "
                f"Optimizer: {optimizer_name} | "
                f"Epochs: {epochs} | "
                f"Accuracy: {accuracy:.4f}"
            )

best_params = max(
    results,
    key=results.get
)

best_accuracy = results[best_params]

print("\nBest Parameters:", best_params)
print("Best Validation Accuracy:", best_accuracy)

best_lr = best_params[0]
best_optimizer = best_params[1]
best_epochs = best_params[2]

print("Best LR:", best_lr)
print("Best Optimizer:", best_optimizer)
print("Best Epochs:", best_epochs)


final_model = CancerModel()

criterion = nn.CrossEntropyLoss()

if best_optimizer == "Adam":

    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_lr
    )

elif best_optimizer == "SGD":

    optimizer = torch.optim.SGD(
        final_model.parameters(),
        lr=best_lr
    )

for epoch in range(best_epochs):

    final_model.train()

    output = final_model(X_train)

    loss = criterion(
        output,
        y_train
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


final_model.eval()

with torch.no_grad():

    output = final_model(X_test)

    prediction = torch.argmax(
        output,
        dim=1
    )

    test_accuracy = (
        prediction == y_test
    ).float().mean().item()

print("\nFinal Test Accuracy:", test_accuracy)
print("\nActual value: ", y_test)
print("\nPrediction: ", prediction)



torch.save(
    final_model.state_dict(),
    "iris_best_model.pth"
)

print("Final model saved!")