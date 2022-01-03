import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchvision import transforms
from torchvision.io import read_image

import pandas as pd


import matplotlib.pyplot as plt

''' 
mnist is the dataset with the corrupted mnist
Args:
    batch_size (int)(optional): batch size of the datasets, default value 8 
'''
def mnist(batch_size: int = 8):

    folder = "./corruptmnist/"
    train_files = [f"train_{i}.npz" for i in range(5)]

    x_train = []
    y_train = []

    for file in train_files:
        with np.load(folder+file) as data:
            x_train.extend(data["images"])
            y_train.extend(data["labels"])
    
    with np.load(folder+"test.npz") as data:
        x_test = data["images"]
        y_test = data["labels"]

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)

    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    norm_transform = transforms.Normalize((0.5,), (0.5,))

    # Create a tensor dataset using x and y, where x is being normalized
    # Then create a dataloader for train and for test
    # Shuffle in train only, as it does not make sense to shuffle test
    train = DataLoader(TensorDataset(norm_transform(x_train), y_train),
                       shuffle=True,
                       batch_size=batch_size)
    
    test = DataLoader(TensorDataset(norm_transform(x_test), y_test),
                      shuffle=False,
                      batch_size=batch_size)

    return train, test

mnist()