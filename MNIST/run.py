import argparse
# from MNIST import MNIST
from MNIST.MNIST import MNIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST')

    parser.add_argument('--optimizer', type=str, default='Adam', dest='optimizer',
                        help='Optimizer for NN: default Adam')
    parser.add_argument('--epochs', type=int, default=2, dest='epochs',
                        help='Number of epochs for training (default 2')
    parser.add_argument('--lr', type=float, default=0.1, dest='lr',
                        help='lr')
    parser.add_argument('--num-layers', type=int, default=1, dest='numlayers',
                        help='numlayers')


    args = parser.parse_args()

    optimizer = args.optimizer
    epochs = args.epochs
    lr = args.lr
    numlayer = args.numlayers

    print('Optimizer: {}\nEpochs: {}\nLearning rate: {}\nNum layers: {} '.format(optimizer, epochs, lr, numlayer))

    a = MNIST(optimizer=optimizer)
    a.train(epochs=epochs)
    _l, ev = a.model.evaluate(a.testDS.x, a.testDS.y_cat)

    print('\ntest_accuracy={}'.format(ev))
#