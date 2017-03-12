# Overview
This code implements a question classifier in singletask learning and multitask learning approach. The primary task dataset is the ['TREC'](http://cogcomp.cs.illinois.edu/Data/QA/QC/) dataset and the secondary task dataset is ['MSMARCO'](http://www.msmarco.org/) dataset. 5 learning algorithms are implemented namely, logistic regression, multilayer perceptron, convolutional neural network, long short term memory network and gated recurrent unit networks.  

#Build dataset
The dataset is already parsed and saved in the directory named 'data' .But it can also be downloaded from the official sites of ['TREC'](http://cogcomp.cs.illinois.edu/Data/QA/QC/) and ['MSMARCO'](http://www.msmarco.org/)
and parsed using the python file 'parse.py'.

#Requirements
- numpy
- pandas
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)
- [gensim](https://radimrehurek.com/gensim/)

The above package if not installed, can be installed using pip command for python : `pip install <package-name>`

#Embedding
The pre-trained word embeddings published by Google was used in this work. The complete word embedding published by Google can be officially downloaded from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

We build the vocablary for the datasets and extract embeddings for only those words in vocabulary by runnning the following code 

>$ python build_vocabulary.py

This creates 4 files namely, vocab_TREC.json and GoogleNews-vectors_TREC.txt which are the vocabulary and filtered embedding file for TREC dataset and  vocab_MS.json and GoogleNews-vectors_MS.txt which are the vocabulary and filtered embedding file for MSMARCO dataset.

To extract the embedding ["Gensim"](https://radimrehurek.com/gensim/) library is used and it requires a decent amount of RAM to run this file (since it uses more than 4GB to load the Google word vector file). So as alternatively, you can download those 4 files from [here](https://drive.google.com/open?id=0B11zdsTNhzfGcW9wdnNLRGUwNmc).Extract the files and place it in the parent directory.

#Build - Singletask
To train the model using normal learing approach - single task learning 

For logisitc and multilayer perceptron algorithms
```sh
python perceptron.py <dataset> <model> 
```
dataset - 'TREC' or 'MS'  
model - 'logistic' or 'MLP'  

For CNN or LSTM or GRU networks
```sh
python run_nnet.py <dataset> <model> 
```
dataset - 'TREC' or 'MS'  
model - 'CNN' or 'LSTM' or 'GRU'  

The trained models are saved in the folder named '<dataset>/<model>' with the embedding file as well. 

#Build - Multitask
Here we load the network trained on MSMARCO dataset and re-train it with TREC data. So to run this the single-task learning on 'MS' data should have been executed earlier and the corresponding model be saved in the default folder. Also you can download a pre-trained model on MSMARCO data from [here]()

```sh
python multitask.py <model> 
```
model - 'logistic' or 'MLP' or 'CNN' or 'LSTM' or 'GRU'

The trained models are saved in the folder named 'MT/<model>'.

