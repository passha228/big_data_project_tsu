import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.metrics import SpecificityAtSensitivity, Recall, CategoricalAccuracy
import os
import cv2

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train / 255
x_test = x_test / 255
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28,1)),
    keras.layers.Dense(128,activation = "relu"),
    keras.layers.Dense(10,activation = "softmax")
])

def FMNIST():
    model.compile(optimizer = tf.keras.optimizers.SGD(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs = 10)


def plot():
    plt.figure()
    plt.imshow(x_train[0])
    plt.colorbar()
    plt.grid(False)

def test():
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy: ', test_acc)
def predict1():
    predictoins = model.predict(x_train)
    print(class_names[np.argmax(predictoins[12])])

def save():
    global model
    nn_trained_path = os.path.join(os.getcwd(), 'FashionMNIST', model.name + '.h5')
    model.save(nn_trained_path)
    
FMNIST()
test()
#predict1()
#save()

sandal = ['sandasl_1.jpg']

path = os.path.join(os.getcwd(), 'sandasl_1.jpg')
predictoins = model.predict(sandal)
print(class_names[np.argmax(predictoins[0])])
def take_image(self, img_path: str):
    """
    img_path: Абсолютный путь
    """
    self.img = cv2.imread(img_path)
    self.img = cv2.resize(self.img, self.input_shape)
    self.img = np.expand_dims(self.img, axis=0)
