import csv
import json
import os
import sys

import numpy as np
import pandas as pd

import utils

np.random.seed(123)

def main() :
    if len(sys.argv) > 2 :
        dataset = sys.argv[1]
        model_name = sys.argv[2]
    else :
        print "model name should be passed.. either logistic or MLP."
        sys.exit(1)

    if dataset in ['TREC', 'MS'] :
        x_train, y_train, x_val, y_val, x_test,y_test =  utils.load_data(dataset)
        embedding_weights = utils.create_embedding(dataset)
    else :
        print "wrong dataset name provided. either TREC or MS"
        sys.exit(1)
    
    vocab_dim = len(embedding_weights.shape[0]) + 1

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

    folder_to_save_files = dataset +'/'+ model_name
    if not os.path.exists(folder_to_save_files):
        os.makedirs(folder_to_save_files)
        
    v_a = 0
    t_a = 0
    if dataset == 'TREC': 
        batch_size = 100
        epoch = 1000
    else : 
        batch_size = 500
        epoch = 50
    np.random.seed(123)
    from keras.callbacks import Callback
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Embedding, Convolution1D, MaxPooling1D, Flatten, LSTM, Merge
    from keras.layers import LSTM, SimpleRNN, GRU
    from keras.callbacks import ModelCheckpoint

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

    model = Sequential()

    if model_name =='CNN' :
        #CNN
        model.add(Embedding(output_dim=300, input_dim=vocab_len, weights=[embedding_weights],trainable=False,input_length=25))
        model.add(Convolution1D(150, 3, activation="relu", border_mode='valid'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(6, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model.summary()

    elif model_name in ['LSTM', 'GRU']:
        #LSTM
        model.add(Embedding(output_dim=300, input_dim=vocab_len, weights=[embedding_weights],trainable=False))
        if model_name == 'LSTM': 
            model.add(LSTM(300,return_sequences=False))
        else:
            model.add(GRU(300,return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(6))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
    else :
        print "Wrong model name..it should be either CNN, LSTM or GRU"
            
    #checkpoint
    filepath = folder_to_save_files + "/model.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,TestCallback((x_test,y_test))]

    open(folder_to_save_files + '/model.json', 'w').write(model.to_json())
    np.save( folder_to_save_files + '/embedding.npy', embedding_weights)

    model.fit(x_train, y_train, validation_data=(x_val,y_val), nb_epoch=epoch, batch_size=batch_size,callbacks=callbacks_list,verbose=1)

    print 'Validation_Acc = ', v_a
    print 'Test_Acc = ', t_a

if __name__ == '__main__':
    main()
