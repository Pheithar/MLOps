import argparse
import os
import sys

import numpy as np
import torch
from model import MyAwesomeModel
from PIL import Image
from torchvision import transforms


class Evaluate(object):
    def __init__(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="")
        parser.add_argument("load_data_from", default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[1:])
        print(args)

        model = MyAwesomeModel()

        to_predict = None
        toTensor = transforms.ToTensor()

        # Folder
        if os.path.isdir(args.load_data_from):
            to_predict = []
            for image in os.listdir(args.load_data_from):
                img = Image.open(os.path.join(args.load_data_from, image))
                img = img.convert("L")
                to_predict.append(toTensor(img))
        # NPY
        else:
            loaded_arr = np.load(args.load_data_from)
            to_predict = [toTensor(x.reshape((28, 28, 1))) for x in loaded_arr]

        model.load_state_dict(torch.load(args.load_model_from))
        model.eval()

        i = 0

        for images in to_predict:
            outputs = model(images)
            out_data = torch.max(outputs, 1)[1].data.numpy()

            for out in out_data:
                print(f"Prediction for input {i}->\t{out}")
                i += 1


if __name__ == "__main__":
    Evaluate()
