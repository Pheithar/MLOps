import matplotlib.pyplot as plt
import seaborn as sns
import torch
from model import MyAwesomeModel
from torch import nn, optim
import hydra
from hydra.utils import get_original_cwd
import random
import numpy as np
import logging
from pytorch_lightning import Trainer, trainer


sns.set()
log = logging.getLogger(__name__)

@hydra.main(config_name="config.yaml", config_path=".")
def main(cfg):
    Train(cfg)

class Train(object):
    """
    Train()

    takes the  learning rate and number of epochs from the standart
    input and train a model that is saved in models/mnisthe/model,
    and displays and saves a figure with the loss function trough
    the training
    """
    def __init__(self, cfg):

        config = cfg.configs

        log.info("Training day and night")
        datapath = get_original_cwd() + config.training.datapath
        epochs =  config.training.epochs
        seed = config.training.epochs
        modelpath = get_original_cwd() + config.training.modelpath

        input_shape = config.model.inputshape
        hidden_layer = config.model.hiddenlayer
        output_shape = config.model.outputshape
        lr = config.model.lr

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        model = MyAwesomeModel(input_shape, hidden_layer, output_shape, lr)
        model.train()



        train_set = torch.load(datapath)

        trainer = Trainer(default_root_dir=modelpath,
                          max_epochs=epochs,
                          limit_train_batches=.05,
                          accelerator="cpu")

        trainer.fit(model, train_set)

        self.trainer = trainer


if __name__ == "__main__":
    main()
