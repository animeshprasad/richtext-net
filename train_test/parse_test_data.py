#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
import os, sys, random
from nltk.tokenize import  WordPunctTokenizer
from collections import Counter
import json, codecs, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import wikipedia
from gensim.summarization.summarizer import summarize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_recall_fscore_support
import json


reload(sys)
sys.setdefaultencoding('utf8')

MAX_LENGTH=30

class DataParser:

    def __init__(self, indir=None, outdir=None):
        self.data_set_citations = pd.read_json('data_set_citations.json', encoding='utf-8')
        self.data_set = pd.read_json('data_sets.json', encoding='utf-8')

        self.data_set_citations.drop(self.data_set_citations.index, inplace=True)
        #self.data_set.drop(self.data_set.index, inplace=True)


        self.publications = pd.read_json(indir+'publications.json', encoding='utf-8')

        self.publications.insert(6, 'full_text', pd.Series)
        self.full_text = self._extract(dir_name=indir+'files/text/')
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

        try:
            self.research_fields = pd.read_csv('saved_research_fields.csv')
            self.research_methods = pd.read_csv('saved_research_methods.csv')
            queries_m = self.research_methods['kw'].tolist()
            queries_f = self.research_fields['kw'].tolist()

        except:
            self.research_fields = pd.read_csv('sage_research_fields.csv')

            sage_research_methods = json.load(open('sage_research_methods.json'))
            research_methods = sage_research_methods['@graph']
            self.research_methods = pd.DataFrame(research_methods)

            queries_m = []
            for index, row in self.research_methods.iterrows():
                try:
                    a = row.loc['skos:altLabel']
                    b = row.loc['skos:prefLabel']
                    if type(a) == type([]):
                        a.append(b)
                    elif type(a) != type({}):
                        a = [b]
                    else:
                        a = [a]
                        a.append(b)
                    j = [items["@value"] for items in a]
                    queries_m.append(j)
                except:
                    queries_m.append([])
            assert (len(queries_m) == len(self.research_methods))

            queries_f = []
            for rf in self.research_fields['L3'].tolist():
                try:
                    #queries_f.append(5 * (' ' + rf))
                    queries_f.append(wikipedia.page(rf).summary)
                except:
                    queries_f.append(5 * (' ' + rf))
            assert (len(queries_f) == len(self.research_fields))


            self.research_fields.insert(len(self.research_fields.columns), 'kw', pd.Series)
            kw = pd.Series(queries_f)
            self.research_fields['kw'] = kw
            self.research_methods.insert(len(self.research_methods.columns), 'kw', pd.Series)
            kw = pd.Series(queries_m)
            self.research_methods['kw'] = kw

            self.research_methods.to_csv('saved_research_methods.csv')
            self.research_fields.to_csv('saved_research_fields.csv')

        for index, row in self.publications.iterrows():
            self.research_methods[row["publication_id"]] = pd.Series(np.random.randn(len(self.research_methods)),
                                                                     index=self.research_methods.index)
            self.research_fields[row["publication_id"]] = pd.Series(np.random.randn(len(self.research_fields)),
                                                                     index=self.research_fields.index)

        docs_m = full_text_series.tolist()
        docs_f = docs_m #[summarize(docs) for docs in docs_m]



        # for i, items in enumerate(queries_f):
        #     if type(items) != type('string'):
        #         queries_f[i] = ''
        # for i, items in enumerate(queries_m):
        #     if type(items) != type('string'):
        #         queries_m[i] = ['']
        #
        #
        # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', lowercase=True)
        # tfidf_matrix = tf.fit(docs_f + queries_f)
        # tfidf_matrix_d = tf.transform(docs_f).todense()
        # tfidf_matrix_q = tf.transform(queries_f).todense()
        # cosine_similarities = linear_kernel(tfidf_matrix_d, tfidf_matrix_q)
        #
        # for i, file_id in zip(self.publications.index, self.publications['publication_id']):
        #     self.research_fields[file_id] = pd.Series(cosine_similarities[i])



        def findall_lower(p, s):
            i = s.lower().find(p.lower())
            while i != -1:
                yield i
                i = s.lower().find(p.lower(), i + 1)

        overall_score = []
        for i, d in enumerate(docs_m):
            score_list = []
            for j, q in enumerate(queries_m):
                score = len([i for qt in eval(q) for i in findall_lower(' '+qt+' ', d)])
                score_list.append(score)
            #self.research_methods[self.publications.iloc[i]['publication_id']] = pd.Series(score_list)
            overall_score.append(score_list)
        
        assert len(overall_score) == len(docs_m)


        research_fields_list = []
        research_methods_list = []

        for i, file_id in zip(self.publications.index, self.publications['publication_id']):
            confidence = max(overall_score[i])
            c= confidence*1.0/ sum(overall_score[i])
            name = eval(self.research_methods.iloc[overall_score[i].index(
                confidence)]["skos:prefLabel"])[u'@value']

            a = {}
            a["publication_id"] = file_id
            a["research_field"] = name
            a["score"] = c
            research_methods_list.append(a)

        with open('../data_submit/output/methods.json ', 'w') as fout2:
            json.dump(research_methods_list, fout2)


        for i, file_id in zip(self.publications.index, self.publications['publication_id']):
            confidence = max(self.research_fields[file_id].tolist())
            name = self.research_fields.iloc[self.research_fields[file_id].tolist().index(
                confidence)]['L3']
            a = {}
            a["publication_id"] = file_id
            a["research_field"] = name
            a["score"] = confidence
            research_fields_list.append(a)

        with open('../data_submit/output/research_fields.json ', 'w') as fout1:
            json.dump(research_fields_list, fout1)



        print (count, "files loaded.")



    def get_train_data_full(self):
        max_length_token = MAX_LENGTH


        with codecs.open(self.outdir + 'golden_test', 'wb') as golden_data:
            for index, row in self.publications.iterrows():
                sample_row = self.publications.loc[self.publications['publication_id'] == row['publication_id']]
                sample_text = sample_row['full_text'].values[0]
                sample_text_tokens = list(WordPunctTokenizer().tokenize(sample_text))

                for splits in range(len(sample_text_tokens) / (max_length_token/2) - 2):
                    if len(sample_text_tokens[splits*(max_length_token/2):(splits+2)*(max_length_token/2)]) != MAX_LENGTH:
                        break
                    #TODO Wrapper over full data reader
                    golden_data.write(
                    str(0) +  ' ' + str(0) +    ' ' + str(0) + ' ' + str(row['publication_id']) +
                    ' ' + ' '.join(sample_text_tokens[splits*(max_length_token/2):(splits+2)*(max_length_token/2)])
                                        + '\n'
                    )



    def _extract(self, dir_name=None, extension='.txt'):
        current_dir = os.getcwd()
        dir_name = current_dir + '/' + dir_name
        full_text = {}
        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with codecs.open(file_name, 'rb') as f:
                    try:
                        lines = f.readlines()
                        #TODO document structure
                        text = ' '.join([s.strip() for s in lines]).decode('utf-8', 'replace')
                        text = re.sub('\d', '0', text)
                        text = re.sub('[^ ]- ', '', text)
                        full_text[item] = text
                    except:
                        pass
        return full_text



class OuputParser:
    #TODO class to produce out json files
    def __init__(self, indir=None, outdir=None):
        self.indir = indir
        self.outdir = outdir



if __name__ == '__main__':
    data_parser = DataParser(indir='../data_submit/input/', outdir='../data/')
    data_parser.get_train_data_full()
