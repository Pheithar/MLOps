# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.io import read_image

''' 
mnist is the dataset with the corrupted mnist
Args:
    batch_size (int)(optional): batch size of the datasets, default value 8 
'''
def mnist(input_filepath, output_filepath, batch_size: int = 8):

    input_folder = f"{input_filepath}/corruptmnist/"
    output_folder = f"{output_filepath}/corruptmnist/"
    train_files = [f"train_{i}.npz" for i in range(5)]

    x_train = []
    y_train = []

    for file in train_files:
        with np.load(input_folder+file) as data:
            x_train.extend(data["images"])
            y_train.extend(data["labels"])
    
    with np.load(input_folder+"test.npz") as data:
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

    torch.save(train, f"{output_folder}train.pt")
    torch.save(test, f"{output_folder}test.pt")

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    mnist(input_filepath, output_filepath)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
