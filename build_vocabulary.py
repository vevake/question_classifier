import numpy as np
from collections import OrderedDict
import re
import pandas as pd
import json
import codecs
from gensim.models import word2vec

def build_dictionary(data_to_dict,file_to_write):   
        word_freqs = OrderedDict()
        for line in data_to_dict:
            words_in = line.strip().split(' ')
            for w in words_in:
                if w not in word_freqs:
                    word_freqs[w] = 0
                word_freqs[w] += 1
        words = word_freqs.keys()
        freqs = word_freqs.values()

        sorted_idx = np.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['<unk>'] = 1
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+2

        with open('%s'%file_to_write, 'wb') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print 'Done'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
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

#vocab for TREC data
trainData = pd.read_csv("data/TREC/train.csv", header=0, delimiter="\t", quoting=3)
trainData['question'] =[clean_str(x) for x in trainData['question']]
build_dictionary(trainData['question'], '/tmp/vocab_train.json')

testData = pd.read_csv("data/TREC/test.csv", header=0, delimiter="\t", quoting=3)
testData['question'] =[clean_str(x) for x in testData['question']]
build_dictionary(testData['question'], '/tmp/vocab_test.json')


#vocab for MSMARCO data
msData = pd.read_csv("data/MS/train.csv", header=0,names=['question', 'label'])
msData['question'] =[clean_str(x) for x in msData['question']]
msData = pd.concat([trainData, msData])
build_dictionary(msData['question'], '/tmp/vocab_ms.json')

files = ['/tmp/vocab_train.json', '/tmp/vocab_ms.json'] 
emb_file = 'GoogleNews-vectors-negative300.bin'
vocab_test = json.loads(open('/tmp/vocab_test.json').read())
model = word2vec.Word2Vec.load_word2vec_format(emb_file, binary=True)

for f in files :
    vocab = json.loads(open(f).read())
    if 'TREC' in f:      
        output_file = 'GoogleNews-vectors_TREC.txt'
        file_to_write = 'data/vocab_TREC.json'
    else :
        output_file = 'GoogleNews-vectors_MS.txt'
        file_to_write = 'data/vocab_MS.json'
    output = codecs.open(output_file, 'w' , 'utf-8')
    found = []
    not_found =[]
    for i,mid in enumerate(vocab):
        vector = list()
        try:
            for dimension in model[mid]:
                vector.append(str(dimension))
            found.append(mid)
            vector_str = " ".join(vector)
            line = mid + "\t"  + vector_str        
            output.write(line + "\n")
        except Exception:
            not_found.append(mid)
    output.close()

    output = codecs.open(output_file, 'a' , 'utf-8')
    found = []
    not_found =[]
    for i,mid in enumerate(vocab_test):
        vector = list()
        try:
            for dimension in model[mid]:
                vector.append(str(dimension))
            found.append(mid)
            vector_str = " ".join(vector)
            line = mid + "\t"  + vector_str
            output.write(line + "\n")
        except Exception:
            not_found.append(mid)
    output.close()

    vocab = set(vocab.keys())
    found = set(found)
    vocab = vocab | found
    vocab = vocab - {'<unk>'}
    worddict = OrderedDict()
    worddict['<unk>'] = 1
    for ii, ww in enumerate(vocab):
        worddict[ww] = ii+2
    with open('%s'%file_to_write, 'wb') as f:
        json.dump(worddict, f, indent=2, ensure_ascii=False)

print 'Embedding extraction completed.'


