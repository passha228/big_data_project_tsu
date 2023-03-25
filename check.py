import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, add, GlobalAveragePooling2D, Dropout, Dense

import os

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = keras.Input(shape=(32, 32, 3), name="img")
x = Conv2D(32, 3, activation="relu")(inputs)
x = Conv2D(64, 3, activation="relu")(x)
block_1_output = MaxPooling2D(3)(x)

x = Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = add([x, block_1_output])

x = Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = add([x, block_2_output])

x = Conv2D(64, 3, activation="relu")(block_3_output)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()

model.compile(optimizer = keras.optimizers.Adam(0.1), loss = 'categorical_crossentropy')
model.fit(x_train, y_train, epochs = 10)
print(model.evaluate(x_test, y_test))

model.save(os.path.join(os.getcwd(), 'dick.h5'))