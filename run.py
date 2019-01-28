import argparse
# from MNIST import MNIST
import itertools
import numpy as np
from MNIST import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')

    parser.add_argument('--optimizer', type=str, default='Adam', dest='optimizer',
                        help='Optimizer for NN: default Adam')
    parser.add_argument('--epochs', type=int, default=2, dest='epochs',
                        help='Number of epochs for training (default 2')

    args = parser.parse_args()

    optimizer = args.optimizer
    epochs = args.epochs

    print('Optimizer: {} \nEpochs: {} '.format(optimizer, epochs, ))

    a = MNIST(optimizer=optimizer)
    a.train(epochs=epochs)
    _l, ev = a.model.evaluate(a.testDS.x, a.testDS.y_cat)

    print('\n\tLABELS: {}\tACCURACY: '.format(a.testDS.y_cat.shape[1]), ev)
