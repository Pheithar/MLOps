from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, input_shape, hidden_layer, output_shape):
        super().__init__()

        self.h1 = nn.Linear(input_shape, hidden_layer[0])
        self.h2 = nn.Linear(hidden_layer[0], hidden_layer[1])

        self.out = nn.Linear(hidden_layer[1], output_shape)

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

    def last_layer_features(self, x):
        x = self.flatten(x)
        x = self.h1(x)
        x = self.sigmoid(x)
        x = self.h2(x)
        x = self.sigmoid(x)

        return x
