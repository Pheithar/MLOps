from pytorch_lightning import LightningModule
from torch import nn, optim


class MyAwesomeModel(LightningModule):
    def __init__(self, input_shape, hidden_layer, output_shape, lr):
        super().__init__()

        self.h1 = nn.Linear(input_shape, hidden_layer[0])
        self.h2 = nn.Linear(hidden_layer[0], hidden_layer[1])

        self.out = nn.Linear(hidden_layer[1], output_shape)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.flatten = nn.Flatten()

        self.criterion = nn.CrossEntropyLoss()

        self.lr = lr

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected input with 3 dimensions")
        if x.shape[1] != 28 or x.shape[2] != 28:
            raise ValueError("Expected input of shape (batch_size, 28, 28)")
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


    def training_step(self, batch, _):
        x, y = batch
        out = self(x)

        y = y.long()

        loss = self.criterion(out, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer