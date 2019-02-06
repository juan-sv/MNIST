import argparse
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import optimizers

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--model', type=str, default='NASNetMobile', dest='model')
    parser.add_argument('--epochs', type=int, default='2', dest='epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', dest='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, dest='lr')

    args = parser.parse_args()

    print(args.model)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if args.model == 'MobileNetV2':
        model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    else:
        model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    if args.optimizer == 'Adam':
        optimizer = optimizers.Adam(lr=args.lr)
    if args.optimizer == 'SGD':
        optimizer = optimizers.SGD(lr=args.lr)
    if args.optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(lr=args.lr)
    if args.optimizer == 'Nadam':
        optimizer = optimizers.Nadam(lr=args.lr)
    if args.optimizer == 'Adamax':
        optimizer = optimizers.Adamax(lr=args.lr)
    if args.optimizer == 'Adadelta':
        optimizer = optimizers.Adadelta(lr=args.lr)
    if args.optimizer == 'Adagrad':
        optimizer = optimizers.Adagrad(lr=args.lr)

    for layer in model.layers:
        layer.trainable = False
    model.trainable = False

    f = Flatten()(model.output)
    output = Dense(10, activation='softmax')(f)

    model = Model(inputs=[model.input], outputs=[output])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    log = '\nepoch {}:\nloss={}\nacc={}\nval_loss={}\nval_acc={}\n'
    cb = [LambdaCallback(on_epoch_end=lambda epoch, logs: print(log.format(epoch,
                                                                           logs['loss'],
                                                                           logs['categorical_accuracy'],
                                                                           logs['val_loss'],
                                                                           logs['val_categorical_accuracy'])))]
    model.fit(x=x_train,
              y=to_categorical(y_train, 10),
              validation_data=(x_test, to_categorical(y_test, 10)),
              epochs=args.epochs,
              callbacks=cb,
              verbose=True)
