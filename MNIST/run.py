import argparse
from MNIST import MNIST
from tensorflow.python.keras.callbacks import LambdaCallback

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MNIST')

    parser.add_argument('--optimizer', type=str, default='Adam', dest='optimizer',
                        help='Optimizer for NN: default Adam')
    parser.add_argument('--epochs', type=int, default=10, dest='epochs',
                        help='Number of epochs for training (default 2')
    parser.add_argument('--lr', type=float, default=0.1, dest='lr',
                        help='lr')
    parser.add_argument('--num-layers', type=int, default=1, dest='numlayers',
                        help='numlayers')
    parser.add_argument('--batch-size', type=int, default=64, dest='batchsize',
                        help='--batch-size')


    args = parser.parse_args()

    optimizer = args.optimizer
    epochs = args.epochs
    lr = args.lr
    numlayer = args.numlayers

    print('Optimizer: {}\nEpochs: {}\nLearning rate: {}\nNum layers: {}\nBatch size: {}'.format(optimizer,
                                                                                                epochs,
                                                                                                lr,
                                                                                                numlayer,
                                                                                                args.batchsize))

    a = MNIST(optimizer=optimizer)

    # epoch 1:
    # loss=0.3
    # recall=0.5
    # precision=0.4
    log = 'epoch {}:\nloss={}\nacc={}\nval_loss={}\nval_acc={}\n'
    cb = [LambdaCallback(on_epoch_end=lambda epoch, logs: print(log.format(epoch,
                                                                           logs['loss'],
                                                                           logs['acc'],
                                                                           logs['val_loss'],
                                                                           logs['val_acc'])))]
    a.train(epochs=epochs, callbacks=cb)

    _l, ev = a.model.evaluate(a.testDS.x, a.testDS.y_cat, verbose=0)

    print('\ntest_accuracy={}'.format(ev))
