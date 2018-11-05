#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import os, sys, random
from nltk.tokenize import  WordPunctTokenizer
from collections import Counter
import json, codecs

reload(sys)
sys.setdefaultencoding('utf8')

class DataParser:

    def __init__(self, outdir=None):
        self.data_set_citations = pd.read_json('data_set_citations.json', encoding='utf-8')
        self.data_set = pd.read_json('data_sets.json', encoding='utf-8')
        self.publications = pd.read_json('publications.json', encoding='utf-8')

        self.publications.insert(6, 'full_text', pd.Series)
        self.full_text = self._extract()
        full_text_series = pd.Series()
        self.outdir = outdir

        count = 0
        for i, file_id in zip(self.publications.index, self.publications['publication_id']):
            try:
                full_text_series.loc[i] = self.full_text[str(file_id) + '.txt']
                count += 1
            except:
                print(str(file_id) + '.txt not found in files')
                pass

        self.publications['full_text'] = full_text_series


    def get_train_data_full(self):
        max_length_token = 500
        pos_neg_sample_ratio = 2 # out of 100

        with codecs.open(self.outdir + 'golden_data', 'wb') as golden_data:
            for index, row in self.data_set_citations.iterrows():
                sample_row = self.publications.loc[self.publications['publication_id'] == row['publication_id']]
                sample_text = sample_row['full_text'].values[0]
                sample_text_tokens = list(WordPunctTokenizer().tokenize(sample_text))
                sample_text_spans = list(WordPunctTokenizer().span_tokenize(sample_text))

                pos_splits = []
                for mention in row['mention_list']:
                    mention_text = mention.encode('utf-8', 'replace')
                    mention_text_spans = list(WordPunctTokenizer().span_tokenize(mention_text))

                    find_index = sample_text.find(mention_text)

                    try:
                        if find_index != -1:
                            mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]
                            #write to training sample pointers here

                            for splits in range(len(sample_text_tokens) / (max_length_token/2) - 2):
                                if sample_text_spans.index(mention_text_spans[0]) > splits*(max_length_token/2) and \
                                  sample_text_spans.index(mention_text_spans[-1]) < (splits+2)*(max_length_token/2):

                                    pos_splits.append(splits)

                                    #TODO Wrapper over full data_b reader
                                    golden_data.write(
                                        #str(row['publication_id']) + ' ' + str(row['data_set_id']) + ' ' +
                                        str(sample_text_spans.index(mention_text_spans[0]) - splits*(max_length_token/2)) +
                                        ' ' + str(sample_text_spans.index(mention_text_spans[-1]) - splits*(max_length_token/2)) +
                                         ' ' + ' '.join(sample_text_tokens[splits*(max_length_token/2):(splits+2)*(max_length_token/2)])
                                        + '\n'
                                    )
                        else:
                            print ('Annotation Error')
                    except:
                        print ('Indexing Logic Error')

                for splits in range(len(sample_text_tokens) / (max_length_token/2) - 2):
                    if splits not in pos_splits and random.randint(0, 100) < pos_neg_sample_ratio:
                        golden_data.write(
                            #str(row['publication_id']) + ' ' + str(0) + ' ' +
                            str(0) +
                            ' ' + str(0) +
                            ' ' + ' '.join(sample_text_tokens[splits * (max_length_token / 2):(splits + 2) * (max_length_token / 2)])
                            + '\n'
                        )


    def get_vocab(self, start_index=2, min_count=10):
        text = ''.join(list(self.publications['full_text'].values))
        all_words = WordPunctTokenizer().tokenize(text + text.lower())
        vocab = Counter(all_words).most_common()
        vocab_out_json = {}
        for items in vocab:
            if items[1] > min_count:
                vocab_out_json[items[0].decode('utf-8', 'replace')] = len(vocab_out_json) + start_index

        print(len(vocab) - len(vocab_out_json), ' words are discarded as OOV')
        print (len(vocab_out_json), ' words are in vocab')

        with codecs.open(self.outdir + 'vocab.json', 'wb') as vocabfile:
            json.dump(vocab_out_json, vocabfile)




    def _extract(self, dir_name='files/text/', extension='.txt'):
        current_dir = os.getcwd()
        dir_name = current_dir + '/' + dir_name
        os.chdir(dir_name)
        full_text = {}
        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with codecs.open(file_name, 'rb') as f:
                    try:
                        lines = f.readlines()
                        #TODO document structure
                        text = ' '.join([s.strip() for s in lines]).decode('utf-8', 'replace')
                        text.replace('- ', '')
                        full_text[item] = text
                    except:
                        pass
        return full_text


if __name__ == '__main__':
    data_parser = DataParser(outdir='/Users/animeshprasad/Desktop/richtext-ptr-net/data/')
    #data_parser.get_vocab()
    data_parser.get_train_data_full()