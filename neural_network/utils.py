import cv2
import numpy as np

import os

root_dir = os.getcwd()
nn_models_dir = os.path.join(root_dir, 'nn_models')
PATH_TO_CONV_NEXT_BASE = os.path.join(nn_models_dir, 'conv_next_base.h5')
PATH_TO_VGG = os.path.join(nn_models_dir, 'vgg.h5')
PATH_TO_INCEPTION_V3 = os.path.join(nn_models_dir, 'inception_v3.h5')
PATH_TO_RESNET_RS350 = os.path.join(nn_models_dir, 'resnet_rs350.h5')

def load_data(path):
    image = cv2.load_image(path)
    imageRgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageRgbResized = cv2.resize(imageRgb, (256, 256))
    imageRgbResized = np.expand_dims(imageRgbResized, axis=0)


def predict(model, value):
    return model.predict(value)
