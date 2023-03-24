#
from keras.models import Sequential, load_model


class NeuralNetwork:
    optimizer = None
    loss = None
    epochs = 0

    def __init__(self, optimizer, loss, epochs):
        """
        Конструктор

        optimizer: мат функция, оптимизатор во время компиляции
        loss: мат. функция потерь
        epochs: кол-во эпох
        """
        self.model = None
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs

    def buildModel(self, inputSize, outputSize):
        """
        Собрать модель

        inputSize: размер изображения в 1-мерной развертке
        outputSize: кол-во классов
        """
        self.model = Sequential()

    def compileModel(self, train_dataset, validation_dataset):
        """
        Обучение модели

        train_datset: датасет для обучения
        validation_datset: датасет для проверки
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model.fit(train_dataset, epochs=self.epochs, validation_data=validation_dataset)

    def saveModelTo(self, path):
        """
        Сохранение обученной модели по указонному пути

        path: полный путь
        """
        self.model.save(path)
    
    def loadModelFromFile(self, path):
        """
        Загрузка модели из указанного файла

        path: полынй путь
        """
        self.model = load_model(path)

    def predict(self, value):
        """
        Предсказание

        value: объект, по которому надо сделать предсказание
        """
        if self.model == None:
            raise RuntimeError("model must be specified")
        return self.model.predict(value)
    
    def validate(self, validation_x, validation_y):
        """
        Проверка нейронной сети

        validation_x: датасет вопросов
        validation_y: датасет ответов
        """
        return self.model.evaluate(validation_x, validation_y)