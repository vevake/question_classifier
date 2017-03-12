#import libraries
import csv
import pandas as pd
import numpy as np

def parse_TREC_data():
    '''Method to parse input data into a csv file of format <id> , <question> , <label>'''

    #parse training data and save in csv file
    with open(dir_path + "TREC/train.txt", "r") as file1, open(dir_path + "TREC/train.csv", "w") as file2:
        writer = csv.writer(file2, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'question', 'label'])
        i = 0
        for line in file1:
            line = line.strip()
            label, question = line.split(' ', 1)
            #print(question,label.split(':')[0])
            writer.writerow([i, question, label.split(':')[0]])
            i += 1

    # parse testing data and save in csv file
    with open(dir_path + "TREC/test.txt", "r") as file1, open(dir_path + "TREC/test.csv", "w") as file2:
        writer = csv.writer(file2, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'question', 'label'])
        i = 0
        for line in file1:
            line = line.strip()
            label, question = line.split(' ', 1)
            # print(question,label.split(':')[0])
            writer.writerow([i, question, label.split(':')[0]])
            i += 1

def parse_MS_data(file_name):
    with open(file_name, 'rb') as f:
        data = f.readlines()

    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    data_df = pd.read_json(data_json_str)
    data_df = data_df[['query','query_type']]
    return data_df

if __name__ == '__main__':
    dir_path = 'data/'

    #parse TREC data
    parse_TREC_data()

    # parse MS data
    trainData = parse_MS_data(dir_path + 'MS/train_v1.1.json')
    testData = parse_MS_data(dir_path + 'MS/test_public_v1.1.json')
    devData = parse_MS_data(dir_path + 'MS/dev_v1.1.json')
    trainData.to_csv(dir_path + 'MS/train.csv', index=False, encoding='utf-8')
    testData.to_csv(dir_path + 'MS/test.csv', index=False, encoding='utf-8')
    devData.to_csv(dir_path + 'MS/dev.csv', index=False, encoding='utf-8')