import argparse
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import MyAwesomeModel
from torch import nn, optim

sns.set()


class Train(object):
    """
    Train()

    takes the  learning rate and number of epochs from the standart
    input and train a model that is saved in models/mnisthe/model,
    and displays and saves a figure with the loss function trough
    the training
    """

    def __init__(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.0001)
        parser.add_argument("--epochs", default=10)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[1:])
        print(args)

        model = MyAwesomeModel()
        model.train()

        train_set = torch.load("data/processed/corruptmnist/train.pt")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        plot_loss = []

        for epoch in range(args.epochs):

            running_loss = 0

            for images, labels in train_set:

                optimizer.zero_grad()

                outputs = model(images)

                labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch: {epoch+1}/{args.epochs}\t Loss: {running_loss/len(train_set):5f}"
            )
            plot_loss.append(running_loss / len(train_set))

        plt.figure(figsize=(12, 8))
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        # Specify the ticks so the figure show all the epochs
        # May be inpractical for longer training
        plt.xticks(range(1, args.epochs + 2))
        plt.ylabel("Loss")
        plt.plot(range(1, args.epochs + 1), plot_loss)
        plt.savefig("reports/figures/training.png")
        plt.close()

        torch.save(model.state_dict(), "models/mnist/model.pt")


if __name__ == "__main__":
    Train()
