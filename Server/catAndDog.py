from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
import matplotlib.pyplot as plt
import os
import cv2
import tkinter.filedialog as tfd
from tensorflow.keras.preprocessing import image_dataset_from_directory

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def create():
    BATCH_SIZE = 32
    IMG_SHAPE = 150

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

    # base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
    # train_dir = os.path.join(base_dir, 'train')
    # validation_dir = os.path.join(base_dir, 'validation')

    # train_cats_dir = os.path.join(train_dir, 'cats')
    # train_dogs_dir = os.path.join(train_dir, 'dogs')
    # validation_cats_dir = os.path.join(validation_dir, 'cats')
    # validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    # num_cats_tr = len(os.listdir(train_cats_dir))
    # num_dogs_tr = len(os.listdir(train_dogs_dir))

    # num_cats_val = len(os.listdir(validation_cats_dir))
    # num_dogs_val = len(os.listdir(validation_dogs_dir))

    # total_train = num_cats_tr + num_dogs_tr
    # total_val = num_cats_val + num_dogs_val
    

    train_dataset = image_dataset_from_directory(
        'C:/Users/Слава/.keras/datasets/cats_and_dogs_filtered/train',
        subset='training',
        validation_split=0.2,
        seed = 42,
        batch_size=256,
        image_size=IMG_SHAPE,
        class_mode = 'binary'
    )

    validation_dataset = image_dataset_from_directory(
        'C:/Users/Слава/.keras/datasets/cats_and_dogs_filtered/validation',
        subset='validation',
        validation_split=0.2,
        seed = 42,
        batch_size=256,
        image_size=IMG_SHAPE,
        class_mode = 'binary'
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # print('Кошек в тестовом наборе данных: ', num_cats_tr)
    # print('Собак в тестовом наборе данных: ', num_dogs_tr)

    # print('Кошек в валидационном наборе данных: ', num_cats_val)
    # print('Собак в валидационном наборе данных: ', num_dogs_val)
    # print('--')
    # print('Всего изображений в тренировочном наборе данных: ', total_train)
    # print('Всего изображений в валидационном наборе данных: ', total_val)

    train_image_generator = ImageDataGenerator(rescale=1./255)
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    # train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #                                                         directory=train_dir,shuffle=True,
    #                                                         target_size=(IMG_SHAPE,IMG_SHAPE),
    #                                                         class_mode='binary')
    # val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    #                                                             directory=validation_dir,
    #                                                             shuffle=False,
    #                                                             target_size=(IMG_SHAPE,IMG_SHAPE),
    #                                                             class_mode='binary')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()

    EPOCHS = 10    
    history = model.fit_generator(
        train_data_gen,
        steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Точность на обучении')
    plt.plot(epochs_range, val_acc, label='Точность на валидации')
    plt.legend(loc='lower right')
    plt.title('Точность на обучающих и валидационных данных')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Потери на обучении')
    plt.plot(epochs_range, val_loss, label='Потери на валидации')
    plt.legend(loc='upper right')
    plt.title('Потери на обучающих и валидационных данных')
    plt.savefig('./foo.png')
    plt.show()

    model.save('./cat_and_dog_e10.h5')

#create()
class_names = ['cat', 'dog']  
model = tf.keras.saving.load_model(os.path.join(os.getcwd(),'cat_and_dog_e30.h5'))
#img = cv2.imread(tfd.askopenfilename())
img = cv2.imread(tfd.askopenfilename(), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
#img /= 255
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (150, 150))
img = np.expand_dims(img, axis=0)

predict = model.predict(img)
print(predict, class_names[np.argmax(predict)])