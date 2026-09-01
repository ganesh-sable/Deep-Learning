import torch 
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4)
        self.layer = nn.Linear(4,1)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer(x)

        return x

model = MyModel()
print(model)
