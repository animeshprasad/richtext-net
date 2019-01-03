import pandas as pd
import os, sys, random
from nltk.tokenize import  WordPunctTokenizer
from collections import Counter
import json, codecs, re

DIR = "../../../train_test/"

data_set_citations = pd.read_json(DIR+'data_set_citations.json', encoding='utf-8')
publications = pd.read_json(DIR+'publications.json', encoding='utf-8')

test_n = 100 

test_ids = random.sample(list(publications['publication_id']), test_n)

def _extract(dir_name='files/text/', extension='.txt'):
        # current_dir = os.getcwd()
        # dir_name = current_dir + '/' + dir_name
        dir_name = DIR + dir_name
        full_text = {}
        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with codecs.open(file_name, 'r') as f:
                    try:
                        lines = f.readlines()
                        #TODO document structure
                        #text = ' '.join([s.strip() for s in lines])
                        text = ' '.join([s.strip() for s in lines])
                        text = re.sub('\d', '0', text)
                        text = re.sub('[^ ]- ', '', text)
                        full_text[item] = text
                    except:
                        pass
        return full_text
full_text = _extract()
full_text_series = pd.Series()
count = 0
for i, file_id in zip(publications.index, publications['publication_id']):
    try:
        full_text_series.loc[i] = full_text[str(file_id)+'.txt']
        count += 1
    except:
        pass

publications.insert(6, 'full_text', pd.Series)
publications['full_text'] = full_text_series

def findall_lower(p, s):
    i = s.lower().find(p.lower())
    while i != -1:
        yield i
        i = s.lower().find(p.lower(), i + 1)



outdir='../../data_doc/'
test_file_ids = open(outdir+'test_file_ids', 'w') 
golden_data = open(outdir+'golden_data', 'w')
test_file = open(outdir+'test_file', 'w')
pub_id = list(data_set_citations['publication_id'])
for t_id in test_ids:
    pub_row = publications.loc[publications['publication_id']==t_id]
    pub_text = pub_row['full_text'].values[0]
    pub_text_tokens = list(WordPunctTokenizer().tokenize(pub_text))
    pub_text_spans = list(WordPunctTokenizer().span_tokenize(pub_text))
    
    res_line = []
    dataset_ids = [pub_id.index(i) for i in pub_id if i==t_id]
    for d_idx in dataset_ids:
        d_row = data_set_citations.loc[d_idx]
        for mention_text in d_row['mention_list']:
            mention_text = re.sub('\d', '0', mention_text)
            mention_text = re.sub('[^ ]- ', '', mention_text)
            mention_text_spans = list(WordPunctTokenizer().span_tokenize(mention_text))
            
            index_finder_lower = findall_lower(mention_text, pub_text)
            found_indices = [idx for idx in index_finder_lower]
            
            for find_index in found_indices:
                try:
                    if find_index != -1:
                        mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]

                        res_line.append((pub_text_spans.index(mention_text_spans[0]), 
                                         pub_text_spans.index(mention_text_spans[-1]), 
                                         d_row['data_set_id'], d_row['publication_id']))
                except:
                    pass
    res_line = list(set(res_line))
    if len(res_line)==0:
        # no mentions as all
        res_line.append((0, 0, 0, t_id))
    test_file_ids.write(str(t_id)+'\n')
    test_file.write(pub_text+'\n')
    i = 0
    for c in res_line:
        if i > 0:
            golden_data.write(' | '+str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
        else:
            golden_data.write(str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
        i+=1
    golden_data.write('\n')

test_file_ids.close()
golden_data.close()
test_file.close()

