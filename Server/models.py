import tensorflow as tf
import keras
import torch
import cv2
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF
import torchvision

from tensorflow.keras import applications

import tkinter.filedialog as tfd

#TODO DEL
from util import take_name_of_predict

class NN:
    model = None
    input_shape = None

    def predict(self):
        return np.argmax(self.model.predict(self.img))
    
    def take_image(self, img_path: str):
        """
        img_path: Абсолютный путь
        """
        self.img = cv2.imread(img_path)
        self.img = cv2.resize(self.img, self.input_shape)
        self.img = np.expand_dims(self.img, axis=0)
    
class DenseNet121(NN):
    def __init__(self) -> None:
        self.model = applications.DenseNet121()
        self.input_shape = (224, 224)

class AlexNet:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

    def take_image(self, img_path: str):
        image = Image.open(img_path)
        image = image.resize([224, 224])

        self.image = TF.to_tensor(image)
        self.image.unsqueeze_(0)

    def predict(self):
        output = self.model(self.image)
        max_arg = torch.argmax(output)
        return max_arg.item()

class VGG19(NN):
    def __init__(self) -> None:
        self.model = applications.VGG19()
        self.input_shape = (224, 224)
