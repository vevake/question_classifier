import csv
import json
import os
import sys

import numpy as np
import pandas as pd

import utils


def main():
    if len(sys.argv) > 2:
        dataset = sys.argv[1]
        model_name = sys.argv[2]

    else:
        print "model name should be passed.. either logistic or MLP."
        sys.exit(1)
    np.random.seed(123)
    x_train, y_train, x_val, y_val, x_test,y_test =  utils.load_data(dataset)
    embedding_weights = utils.create_embedding(dataset)

    vocab_dim = embedding_weights.shape[1]

    train = np.zeros((len(x_train), vocab_dim))
    for i,x in enumerate(x_train):
        a = np.zeros((len(x),vocab_dim))
        for j,word in enumerate(x):
            a[j,:] = embedding_weights[word]
        train[i] = a.mean(axis=0)

    val = np.zeros((len(x_val),vocab_dim))
    for i,x in enumerate(x_val):
        a = np.zeros((len(x),vocab_dim))
        for j,word in enumerate(x):
            a[j,:] = embedding_weights[word]
        val[i] = a.mean(axis=0)

    test = np.zeros((len(x_test),vocab_dim))
    for i,x in enumerate(x_test):
        a = np.zeros((len(x),vocab_dim))
        for j,word in enumerate(x):
            a[j,:] = embedding_weights[word]
        test[i] = a.mean(axis=0)
    
    folder_to_save_files = dataset + '/' + model_name
    if not os.path.exists(folder_to_save_files):
        os.makedirs(folder_to_save_files)

    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, nb_classes=6)
    y_val = np_utils.to_categorical(y_val, nb_classes=6)
    y_test = np_utils.to_categorical(y_test, nb_classes=6)
    
    np.random.seed(123)
    from keras.callbacks import Callback
    from keras.models import Sequential
    from keras.layers.core import Reshape
    from keras.layers import Input,Dense, Dropout, Activation, Embedding,Merge
    from keras.layers import LSTM, SimpleRNN, GRU
    from keras.callbacks import ModelCheckpoint

    import tensorflow as tf
    tf.python.control_flow_ops = tf

    class TestCallback(Callback):
        def __init__(self, test_data):
            self.test_data = test_data
            self.val_acc = 0

        def on_epoch_end(self, epoch, logs={}):
            if self.val_acc < logs.get('val_acc'):
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
    callbacks_list = [checkpoint,TestCallback((test,y_test))]


    if model_name == 'logistic' :
    # Logistic layer
        model=Sequential()
        model.add(Dense(6, input_dim=vocab_dim, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

    elif model_name == 'MLP' :
        #Multi layer perceptron
        model = Sequential()
        model.add(Dense(128, input_dim=vocab_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
    
    else :
        print "wrong model name provided"
        sys.exit(1)
        
    open(folder_to_save_files + '/model.json', 'w').write(model.to_json())
    np.save(folder_to_save_files + '/embedding.npy', embedding_weights)    

    model.fit(train,y_train,validation_data=[val, y_val], nb_epoch=1000, batch_size=100,callbacks=callbacks_list,verbose=1)
    print 'Validation_Acc = ', v_a
    print 'Test_Acc = ', t_a

if __name__ == '__main__':
    np.random.seed(123)
    main()
