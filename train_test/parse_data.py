#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import os, sys
from nltk.tokenize import  WordPunctTokenizer
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf8')

class DataParser:

    def __init__(self, outdir=None):
        self.data_set_citations = pd.read_json('data_set_citations.json')
        self.data_set = pd.read_json('data_sets.json')
        self.publications = pd.read_json('publications.json')

        self.publications.insert(6, 'full_text', pd.Series)
        self.full_text = self._extract()
        full_text_series  = pd.Series()
        self.outdir = outdir

        count = 0
        for i, file_id in zip(self.publications.index, self.publications['publication_id']):
            try:
                full_text_series.loc[i] =  self.full_text[str(file_id) + '.txt']
                count +=1
            except:
                print(str(file_id) + '.txt not found in files')
                pass

        self.publications['full_text'] = full_text_series


    def get_train_data_full(self):
        with open(self.outdir + 'golden_data', 'w') as golden_data:
            for index, row in self.data_set_citations.iterrows():
                sample_row = self.publications.loc[self.publications['publication_id'] == row['publication_id']]
                sample_text = sample_row['full_text'].values[0]
                sample_text_spans = list(WordPunctTokenizer().span_tokenize(sample_text))

                for mention in row['mention_list']:
                    mention_text = mention.encode('utf-8')
                    mention_text_spans = list(WordPunctTokenizer().span_tokenize(mention_text))

                    find_index = sample_text.find(mention_text)
                    try:
                        if find_index != -1:
                            mention_text_spans = [ (indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]
                            #write to training sample pointers here
                            #TODO Wrapper over full data reader
                            golden_data.write(sample_text_spans.index(mention_text_spans[0]) +
                                              ' ' + sample_text_spans.index(mention_text_spans[-1]) +
                                              ' ' + sample_text)
                        else:
                            print ('Annotation Error')
                    except:
                        print ('Indexing Logic Error')


    def get_vocab(self, start_index=2, min_count=10):
        text = ''.join(list(self.publications['full_text'].values))
        all_words = WordPunctTokenizer().tokenize(text + text.lower())
        vocab = Counter(all_words).most_common()
        vocab_out_json = {}
        for items in vocab:
            if items[1] > min_count:
                vocab_out_json[len(vocab_out_json)+start_index] = items[0]

        print(len(vocab) - len(vocab_out_json), ' words are discarded as OOV')
        print (len(vocab_out_json), ' words are in vocab' )
        with open (self.outdir + 'vocab.json', 'w') as vocabfile:
            vocabfile.write(vocab_out_json)


    def _print(self):
        print(self.data_set_citations)
        print (self.data_set)
        print(self.publications)

    def _extract(self, dir_name='files/text/', extension='.txt'):
        current_dir = os.getcwd()
        dir_name = current_dir + '/' + dir_name
        os.chdir(dir_name)

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


if __name__ == '__main__':
    data_parser = DataParser(outdir='../data/')
    data_parser.get_vocab()
    data_parser.get_train_data_full()