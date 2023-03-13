#
from keras.models import Sequential, load_model


class NeuralNetwork:
    optimizer = None
    loss = None
    epochs = 0

    def __init__(self, optimizer, loss, epochs):
        self.model = None
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs

    def buildModel(self, inputSize, outputSize):
        self.model = Sequential()

    def compileModel(self, train_dataset, validation_dataset):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        self.model.fit(train_dataset, epochs=self.epochs, validation_data=validation_dataset)

    def saveModelTo(self, path):
        self.model.save(path)
    
    def loadModelFromFile(self, path):
        self.model = load_model(path)

    def predict(self, value):
        if self.model == None:
            raise RuntimeError("model must be specified")
        return self.model.predict(value)
    
    def validate(self, validation_x, validation_y):
        return self.model.evaluate(validation_x, validation_y)