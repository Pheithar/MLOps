from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.h1 = nn.Linear(784, 512)
        self.h2 = nn.Linear(512, 256)

        self.out = nn.Linear(256, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.h1(x)
        x = self.sigmoid(x)
        x = self.h2(x)
        x = self.sigmoid(x)
        x = self.out(x)
        x = self.softmax(x)


        return x