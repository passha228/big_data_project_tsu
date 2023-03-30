import tensorflow as tf
import keras


from tensorflow.keras import applications, Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10

import os
import subprocess


# https://tech.bertelsmann.com/en/blog/articles/convnext
# UNIT_TESTS_DONE
def create_ConvNeXtBase() -> keras.Model:
    model = applications.ConvNeXtBase(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'relu')(x)

    return keras.Model(inputs = model.input, outputs = x)


# UNIT_TESTS_DONE
def create_vgg() -> keras.Model:
    model = applications.VGG16(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'relu')(x)

    return keras.Model(inputs = model.input, outputs = x)


# UNIT_TESTS_DONE
def create_InceptionV3() -> keras.Model:
    model = applications.VGG16(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'relu')(x)

    return keras.Model(inputs = model.input, outputs = x)


# UNIT_TESTS_DONE
def create_ResNetRS350() -> keras.Model:
    model = applications.ResNetRS350(classes = 10, input_shape = (32, 32, 3), include_top = False)
    x = Flatten()(model.output)
    x = Dense(10, activation = 'relu')(x)

    return keras.Model(inputs = model.input, outputs = x)


class Training:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255
        self.y_train = keras.utils.to_categorical(self.y_train, 10)
        self.y_test = keras.utils.to_categorical(self.y_test, 10)

    def train_save(self, model):
        # Обучение
        model.compile(optimizer = keras.optimizers.Adam(0.1), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        with tf.device('/GPU:0'):
            model.fit(self.x_train, self.y_train, epochs = 10)

        # сохранение
        nn_trained_path = os.path.join(os.getcwd(), 'nn_models', model.name + 'h5')
        model.save(nn_trained_path)
    
    def train_save_all(self):
        self.train_save(create_vgg())
        self.train_save(create_InceptionV3())
        self.train_save(create_ResNetRS350())
        self.train_save(create_ConvNeXtBase())

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Разрешаем рост памяти GPU по мере необходимости
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Ошибка при конфигурации GPU
            print(e)
    print(gpus)
    #trainer = Training()
    #trainer.train_save_all()