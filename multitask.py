import os
import sys

import numpy as np

import utils

np.random.seed(123)

def main():
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        print "model name should be passed.. either logistic or MLP."
        sys.exit(1)
    dataset = 'TREC'
    x_train, y_train, x_val, y_val, x_test, y_test = utils.load_data(dataset, 'MT')

    if model_name in ['CNN', 'LSTM', 'GRU']:
        #use padding to input to neural net
        max_len = 25
        from keras.preprocessing import sequence
        from keras.utils import np_utils
        x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        y_train = np_utils.to_categorical(y_train, nb_classes=6)

        x_val = sequence.pad_sequences(x_val, maxlen=max_len)
        y_val = np_utils.to_categorical(y_val, nb_classes=6)

        x_test = sequence.pad_sequences(x_test, maxlen=max_len)
        y_test = np_utils.to_categorical(y_test, nb_classes=6)

    elif model_name in ['logistic', 'MLP']:
        embedding_weights = np.load('MS/' + model_name + '/embedding.npy')
        vocab_dim = embedding_weights.shape[1]

        train = np.zeros((len(x_train), vocab_dim))
        for i, x in enumerate(x_train):
            a = np.zeros((len(x), vocab_dim))
            for j, word in enumerate(x):
                a[j, :] = embedding_weights[word]
            train[i] = a.mean(axis=0)
        x_train = train

        val = np.zeros((len(x_val), vocab_dim))
        for i, x in enumerate(x_val):
            a = np.zeros((len(x), vocab_dim))
            for j, word in enumerate(x):
                a[j, :] = embedding_weights[word]
            val[i] = a.mean(axis=0)
        x_val = val

        test = np.zeros((len(x_test), vocab_dim))
        for i, x in enumerate(x_test):
            a = np.zeros((len(x), vocab_dim))
            for j, word in enumerate(x):
                a[j, :] = embedding_weights[word]
            test[i] = a.mean(axis=0)
        x_test = test

        from keras.utils import np_utils
        y_train = np_utils.to_categorical(y_train, nb_classes=6)
        y_val = np_utils.to_categorical(y_val, nb_classes=6)
        y_test = np_utils.to_categorical(y_test, nb_classes=6)

    else:
        print "enter valid model name."
        sys.exit(1)

    folder_to_save_files = 'MT/' + model_name
    if not os.path.exists(folder_to_save_files):
        os.makedirs(folder_to_save_files)

    model_to_load = 'MS/' + model_name + '/model.hdf5'

    np.random.seed(123)
    from keras.callbacks import Callback, ModelCheckpoint

    import tensorflow as tf
    tf.python.control_flow_ops = tf

    class TestCallback(Callback):
        def __init__(self, test_data):
            self.test_data = test_data
            self.val_acc = 0

        def on_epoch_end(self, epoch, logs={}):
            if self.val_acc < logs.get('val_acc') :
                self.val_acc = logs.get('val_acc')
                x, y = self.test_data
                loss, acc = self.model.evaluate(x, y, verbose=0)
                global t_a
                global v_a
                v_a = self.val_acc
                t_a = acc
                print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

    #checkpoint
    filepath = folder_to_save_files + "/model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((x_test, y_test))]

    from keras.models import load_model
    model = load_model(model_to_load)

    model.fit(x_train, y_train, validation_data=[x_val, y_val], nb_epoch=1000, batch_size=100, callbacks=callbacks_list, verbose=1)

    print 'Validation_Acc = ', v_a
    print 'Test_Acc = ', t_a

if __name__ == '__main__':
    main()
