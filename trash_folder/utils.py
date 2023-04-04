import cv2
import numpy as np

import os

root_dir = os.getcwd()
nn_models_dir = os.path.join(root_dir, 'nn_models')

# Пути до предобученных моделей
PATH_TO_CONV_NEXT_BASE = os.path.join(nn_models_dir, 'conv_next_base.h5')
PATH_TO_VGG = os.path.join(nn_models_dir, 'vgg.h5')
PATH_TO_INCEPTION_V3 = os.path.join(nn_models_dir, 'inception_v3.h5')
PATH_TO_RESNET_RS350 = os.path.join(nn_models_dir, 'resnet_rs350.h5')

IMG_SIZE = (32, 32)

def load_data(path):
    image = cv2.load_image(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    return image


def predict(model, value):
    return model.predict(value)
