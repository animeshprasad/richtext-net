import pandas as pd
import os, sys
from collections import Counter
import numpy as np

# change this dir if needed
data_dir = "../train_test/"
data_set_citations = pd.read_json(data_dir+'data_set_citations.json')
data_set = pd.read_json(data_dir+'data_sets.json')
publications = pd.read_json(data_dir+'publications.json')

data_set_citations.to_csv("data_set_citations.csv")
data_set.to_csv("data_set.csv")

def extract(dir_name='../train_test/files/text/', extension='.txt'):
        full_text = {}

        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with open(file_name, 'r') as f:
                    try:
                        lines = f.readlines()
                        text = ' '.join([s.strip() for s in lines])
                        full_text[item] = text
                    except:
                        pass
        return full_text

full_text = extract()

def split_string(string, n=100):
    str_list = string.strip().split()
    str_len = len(str_list)
    each_len = str_len // n
    res = []
    for i in range(n-1):
        res.append(' '.join(str_list[i*each_len : (i+1)*each_len]))
    res.append(' '.join(str_list[(n-1)*each_len : ]))
    return res

full_text_series  = pd.Series()

count = 0
for i, file_id in zip(publications.index, publications['publication_id']):
    try:
        full_text_series.loc[i] =  split_string(full_text[str(file_id) + '.txt'])
        count +=1
    except:
        print(str(file_id) + '.txt not found in files')
        pass

publications['full_text'] = full_text_series
publications.to_csv("publications.csv")

def get_train_data_full():
    f_sentences = open("sentences", 'w')  
    f_labels = open("labels", 'w')
    has_data_count = 0
    no_data_count = 0
    
    for index, row in data_set_citations.iterrows():
        sample_row = publications.loc[publications['publication_id'] == row['publication_id']]
        sample_text = sample_row['full_text'].item()  # a list of sentences 
        print (index)
        
        for sentence in sample_text:
            f_sentences.write(sentence+'\n')
            has_data = False
            for mention in row["mention_list"]:
                find_index = sentence.find(mention)
                if find_index != -1:
                    f_labels.write(str(row['data_set_id']) + '\n')
                    has_data = True
                    has_data_count += 1
                    break
            if not has_data:
                f_labels.write('0\n')
                no_data_count += 1
    
    print ("Number of positive labels: ", has_data_count)
    print ("Number of negative labels: ", no_data_count)
    
    f_sentences.close()
    f_labels.close()

get_train_data_full()






