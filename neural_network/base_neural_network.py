#
from keras.models import Sequential, load_model


class NeuralNetwork:
    def __init__(self, optimizer = None, loss = None, epochs = 0):
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

    def compileModel(self, train_dataset, validation_dataset):
        """
        Обучение модели

        train_datset: датасет для обучения
        validation_datset: датасет для проверки
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        return self.model.fit(train_dataset, validation_dataset, epochs=self.epochs)

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