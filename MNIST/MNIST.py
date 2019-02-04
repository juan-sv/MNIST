import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model  # , Sequential, load_model
# from keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.datasets import mnist
# from keras.activations import softmax
# from keras.callbacks import TensorBoard
# from time import time
from tensorflow.keras.callbacks import TensorBoard


class MNIST:
    class DataSet:
        def __init__(self):
            self.x = None
            self.y = None
            self.y_cat = None

    def __init__(self, optimizer=None):
        self.__getData__()
        if optimizer:
            self.__get_model__(optimizer)
        else:
            self.__get_model__()
        self.log = ''

    def __getData__(self):
        self.trainDS = self.DataSet()
        self.testDS = self.DataSet()
        (self.trainDS.x, self.trainDS.y), (self.testDS.x, self.testDS.y) = mnist.load_data()
        self.trainDS.y_cat = to_categorical(self.trainDS.y)
        self.testDS.y_cat = to_categorical(self.testDS.y)

    def __get_model__(self, optimizer: str = 'Adam'):
        i = Input(shape=(28, 28,))
        h1 = Dense(370, activation='tanh')(i)
        h2 = Dense(37, activation='tanh')(h1)
        flat = Flatten(input_shape=(28, 37))(h2)
        o = Dense(self.trainDS.y_cat.shape[1], activation='sigmoid')(flat)
        self.model = Model(inputs=i,
                           outputs=o)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=1):

        tb = None
        # tb = [TensorBoard(log_dir="C:/Temp/logs/{}".format(log))]
        #                          histogram_freq=0,
        #                          batch_size=32,
        #                          write_graph=True,
        #                          write_grads=True,
        #                          write_images=True)
        #                          embeddings_freq=0,
        #                          embeddings_layer_names=None,
        #                          embeddings_metadata=None)

        self.model.fit(x=self.trainDS.x,
                       y=self.trainDS.y_cat,
                       validation_data=(self.testDS.x, self.testDS.y_cat),
                       epochs=epochs,
                       callbacks=tb)
