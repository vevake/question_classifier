import numpy as np
import csv
import pandas as pd
import json
import sys

import re
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def load_data(dataset,task = 'ST'):
    if dataset == 'TREC':
        trainData = pd.read_csv("data/TREC/train.csv", header=0, delimiter="\t", quoting=3)
        testData = pd.read_csv("data/TREC/test.csv", header=0, delimiter="\t", quoting=3)

        trainData = trainData.iloc[np.random.permutation(len(trainData))]
        val_len = int(round(len(trainData)*0.15))
        train = trainData.head(len(trainData)-val_len)
        val = trainData.tail(val_len)
        if task == 'ST':
            vocab = json.loads(open('vocab_TREC.json').read())
        elif task == 'MT':
            vocab = json.loads(open('vocab_MS.json').read())
        class_ = {'ABBR': 0, 'DESC': 1, 'ENTY': 2, 'HUM': 3, 'LOC' : 4, 'NUM': 5 }
    elif dataset == 'MS':
        train = pd.read_csv('data/MS/train.csv',header=0,names=['question', 'label'])
        val = pd.read_csv('data/MS/dev.csv',header=0,names=['question', 'label'])
        testData = pd.read_csv('data/MS/test.csv',header=0,names=['question', 'label'])
        vocab = json.loads(open('vocab_MS.json').read())
        class_ = {'description': 1, 'entity': 2, 'person': 3, 'location' : 4, 'numeric': 5 }
    else:
        print "wrong dataset"
        sys.exit(1)

    train['question'] = [clean_str(x) for x in train['question']]
    val['question'] = [clean_str(x) for x in val['question']]
    testData['question'] = [clean_str(x) for x in testData['question']]

    x_train = []
    for ques in train['question']:
        s = ques.strip().split()
        x_train.append([vocab[x] if x in vocab else 1 for x in s])
    y_train = []
    for label in train['label']:
        y_train.append([class_[label]])

    x_val = []
    for ques in val['question']:
        s = ques.strip().split()
        x_val.append([vocab[x] if x in vocab else 1 for x in s])
    y_val = []
    for label in val['label']:
        y_val.append([class_[label]])

    x_test = []
    for ques in testData['question']:
        s = ques.strip().split()
        x_test.append([vocab[x] if x in vocab else 1 for x in s])
    y_test = []
    for label in testData['label']:
        y_test.append([class_[label]])

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_embedding(dataset,task='ST'):
    embeddings_index = {}
    if dataset == 'TREC' and task == 'ST':
        word_vec = 'GoogleNews-vectors_TREC.txt'
        vocab = json.loads(open('vocab_TREC.json').read())
    elif dataset == 'MS' or task == 'MT':
        word_vec = 'GoogleNews-vectors_MS.txt'
        vocab = json.loads(open('vocab_MS.json').read())
    else:
        print 'error loading the embedding.'
        sys.exit(1)

    f = open(word_vec)
    for emb in f:
        values = emb.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()

    vocab_dim = 300 # dimensionality of your word vectors
    vocab_len = len(vocab)+1
    """
    For words that does not have pretrained word embeddings.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones

    As suggested by YoonKim in "Convolutional Neural Networks for Sentence Classification"
    https://github.com/yoonkim/CNN_sentence
    """
    embedding_weights = np.zeros((vocab_len,vocab_dim))
    for word,index in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None :       
            embedding_weights[index,:] = embedding_vector
        else :
            embedding_weights[index,:] = np.random.uniform(-0.25,0.25,vocab_dim)
    
    return embedding_weights