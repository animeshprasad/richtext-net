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


        self.research_fields = pd.read_csv('sage_research_fields.csv')

        sage_research_methods = json.load(open('sage_research_methods.json'))
        research_methods = sage_research_methods['@graph']
        self.research_methods = pd.DataFrame(research_methods)


        for index, row in self.publications.iterrows():
            self.research_methods[row["publication_id"]] = pd.Series(np.random.randn(len(self.research_methods)),
                                                                     index=self.research_methods.index)
            self.research_fields[row["publication_id"]] = pd.Series(np.random.randn(len(self.research_fields)),
                                                                     index=self.research_fields.index)

        docs_m = full_text_series.tolist()
        docs_f = [summarize(docs) for docs in docs_m]

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
                #queries_f.append(rf)
                queries_f.append(wikipedia.page(rf).summary)
            except:
                queries_f.append(5 * (' ' + rf))
        assert (len(queries_f) == len(self.research_fields))


        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english', lowercase=True)
        tfidf_matrix = tf.fit_transform(docs_f + queries_f)
        tfidf_matrix_d = tf.fit(docs_f)
        tfidf_matrix_q = tf.fit(queries_f)
        cosine_similarities = linear_kernel(tfidf_matrix_d, tfidf_matrix_q).flatten()

        #TODO assign to the DF

        def findall_lower(p, s):
            i = s.lower().find(p.lower())
            while i != -1:
                yield i
                i = s.lower().find(p.lower(), i + 1)

        for d in docs_m:
            score = 0
            for q in queries_m:
                score += len([i for i in findall_lower(q, d)])

        # TODO assign to the DF



        print (count, "files loaded.")



    def get_train_data_full(self, neg_ratio=1):
        max_length_token = MAX_LENGTH
        pos_neg_sample_ratio = neg_ratio #5 out of 1000

        pos_count = 0
        neg_count = 0

        with codecs.open(self.outdir + 'golden_data', 'wb') as golden_data:
            for index, row in self.data_set_citations.iterrows():
                sample_row = self.publications.loc[self.publications['publication_id'] == row['publication_id']]
                sample_text = sample_row['full_text'].values[0]
                sample_text_tokens = list(WordPunctTokenizer().tokenize(sample_text))
                sample_text_spans = list(WordPunctTokenizer().span_tokenize(sample_text))

                pos_splits = []
                for mention in row['mention_list']:
                    mention_text = mention.encode('utf-8', 'replace')
                    mention_text = re.sub('\d', '0', mention_text)
                    mention_text = re.sub('[^ ]- ', '', mention_text)
                    mention_text_spans = list(WordPunctTokenizer().span_tokenize(mention_text))


                    def findall(p, s):
                        '''Yields all the positions of
                        the pattern p in the string s.'''
                        i = s.find(p)
                        while i != -1:
                            yield i
                            i = s.find(p, i + 1)

                    def findall_lower(p, s):
                        i = s.lower().find(p.lower())
                        while i != -1:
                            yield i
                            i = s.lower().find(p.lower(), i + 1)


                    index_finder = findall(mention_text, sample_text)
                    index_finder_lower = findall_lower(mention_text, sample_text)

                    found_indices = [idx for idx in index_finder]
                    found_indices_lower = [idx for idx in index_finder_lower]

                    all_found_indices = list(set(found_indices + found_indices_lower))



                    for find_index in all_found_indices:
                      try:
                        if find_index != -1:
                            mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]
                            #write to training sample pointers here

                            for splits in range(len(sample_text_tokens) / (max_length_token/2) - 2):
                                if sample_text_spans.index(mention_text_spans[0]) > splits*(max_length_token/2) and \
                                  sample_text_spans.index(mention_text_spans[-1]) < (splits+2)*(max_length_token/2):

                                    pos_splits.append(splits)
                                    pos_count += 1

                                    #TODO Wrapper over full data reader
                                    golden_data.write(
                                        str(sample_text_spans.index(mention_text_spans[0]) - splits*(max_length_token/2)) +
                                        ' ' + str(sample_text_spans.index(mention_text_spans[-1]) - splits*(max_length_token/2)) +
                                        ' ' + str(row['data_set_id']) + ' ' + str(row['publication_id']) +
                                         ' ' + ' '.join(sample_text_tokens[splits*(max_length_token/2):(splits+2)*(max_length_token/2)])
                                        + '\n'
                                    )
                        else:
                            print ('Annotation Error: Annotated gold standards not correct')
                            pass
                      except:
                        #print ('Indexing Logic Error: Some corner index case missed while parsing')
                        pass

                        # print mention_text_spans
                        # print (find_index)
                        # print (sample_text[mention_text_spans[0][0]: mention_text_spans[-1][1]])
                        # print (mention_text)
                        #
                        # for i in findall_lower(mention_text, sample_text):
                        #     print i
                        #
                        # print (find_index, sample_text_spans.index(mention_text_spans[0]),  sample_text_spans.index(mention_text_spans[-1]),
                        #        splits * (max_length_token), (splits+1)*(max_length_token))
                        #
                        # print mention_text, row['publication_id']
                        # raw_input()


                for splits in range(len(sample_text_tokens) / (max_length_token / 2) - 2):
                    if splits not in pos_splits and random.randint(0, 1000) < pos_neg_sample_ratio:
                        golden_data.write(
                            str(0) + ' ' + str(0) +
                            ' ' + str(0) + ' ' + str(row['publication_id']) +
                            ' ' + ' '.join(sample_text_tokens[splits * (max_length_token / 2):(splits + 2) * (
                                                max_length_token / 2)])
                            + '\n'
                        )

                        neg_count += 1

        print (pos_count, "mentions added.")
        print (neg_count, "no mentions added.")

        with codecs.open(self.outdir + 'golden_data', 'rb') as golden_data, \
            codecs.open(self.outdir + 'train.txt', 'wb') as train_split, \
            codecs.open(self.outdir + 'validate.txt', 'wb') as validate_split, \
            codecs.open(self.outdir + 'test.txt', 'wb') as test_split:
            all_lines = golden_data.readlines()
            for i, line in enumerate(all_lines):
                if i%9 == 0:
                    validate_split.write(line)
                elif i%10 == 0:
                    test_split.write(line)
                else:
                    train_split.write(line)




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


class DummydataParser:

    def __init__(self):
        pass

    def _extract(self, dir_name='files/text/', extension='.txt'):
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

    def get_predict_split(self):
        self.data_set_citations = pd.read_json('data_set_citations.json', encoding='utf-8')
        self.data_set = pd.read_json('data_sets.json', encoding='utf-8')
        self.publications = pd.read_json('publications.json', encoding='utf-8')

        self.data_set_citations.drop(self.data_set_citations.index, inplace=True)
        self.data_set.drop(self.data_set.index, inplace=True)
        self.publications.drop(self.publications.index, inplace=True)


        self.publications.insert(6, 'full_text', pd.Series)
        self.full_text = self._extract()
        full_text_series = pd.Series()

        # count = 0
        # for i, file_id in zip(self.publications.index, self.publications['publication_id']):
        #     try:
        #         full_text_series.loc[i] = self.full_text[str(file_id) + '.txt']
        #         count += 1
        #     except:
        #         print(str(file_id) + '.txt not found in files')
        #         pass
        #
        # self.publications['full_text'] = full_text_series
        #
        # for splits in range(len(sample_text_tokens) / (max_length_token) - 1):
        #     if splits not in pos_splits and random.randint(0, 1000) < pos_neg_sample_ratio:
        #         golden_data.write(
        #             str(0) + ' ' + str(0) +
        #             ' ' + str(0) + ' ' + str(row['publication_id']) +
        #             ' ' + ' '.join(sample_text_tokens[splits * (max_length_token):(splits + 1) * (max_length_token)])
        #             + '\n'
        #         )


class OuputEvaluator:

    def __init__(self, outdir=None):
        self.outdir = outdir

    # TODO: doc level evaluation here

    def generate_report(self, ground_truth='test.txt', model_out='model_test.txt'):
        with codecs.open(self.outdir + ground_truth, 'rb') as ground_split, \
            codecs.open(self.outdir + model_out, 'rb') as model_out_split:
            true_lines = ground_split.readlines()
            predict_lines = model_out_split.readlines()

            def compute_partial(s1,s2,e1,e2):
                tp = min(max(s1,s2) - min(e1, e2), 0)
                total_predicted = (e1-s1) + (e2-s2)
                try:
                    partial_accuracy = float(tp)/total_predicted
                except:
                    partial_accuracy = 0
                try:
                    partial_precision = float(tp)/ (e2-s2)
                except:
                    partial_precision = 0
                try:
                    partial_recall = float(tp)/ (e1-s1)
                except:
                    partial_recall = 0
                try:
                    partial_fscore = 2 * (partial_precision * partial_recall) / (partial_precision + partial_recall)
                except:
                    partial_fscore = 0
                return partial_accuracy, partial_precision, partial_recall, partial_fscore

            def match_report():
                true_mention = []
                predict_mention = []
                true_dataset = []
                predict_dataset = []

                partial_accuracy = []
                partial_precision = []
                partial_recall = []
                partial_fscore = []

                for trues, predicts in zip(true_lines, predict_lines):
                    trues = [int(t) for t in trues.split()[:4]]
                    predicts = [int(t) for t in predicts.split()[:4]]
                    print (trues, predicts)

                    if predicts[0] != -1:
                        if trues[0] == predicts[0] and trues[1] == predicts[1]:
                            true_mention.append(1)
                            predict_mention.append(1)
                        else:
                            predict_mention.append(0)

                        a,b,c,d = compute_partial(trues[0],predicts[0],trues[1],predicts[1])
                        partial_accuracy.append(a)
                        partial_precision.append(b)
                        partial_recall.append(c)
                        partial_fscore.append(d)

                    if predicts[2] != -1:
                        true_dataset.append(trues[2])
                        predict_dataset.append(predicts[2])

                P, R, F, S = precision_recall_fscore_support(true_mention, predict_mention)
                print(P, R, F)
                #print (P[1], R[1], F[1])

                P, R, F, S = precision_recall_fscore_support(true_dataset, predict_dataset)
                print(np.average(P), np.average(R), np.average(F))

                A, P, R, F = np.average(partial_accuracy), np.average(partial_precision), \
                             np.average(partial_recall), np.average(partial_fscore)
                print (P, R, F, A)

            match_report()




class OuputParser:
    #TODO class to produce out json files
    def __init__(self, outdir=None):
        self.outdir = outdir



if __name__ == '__main__':
    data_parser = DataParser(outdir='../data/')
    data_parser.get_train_data_full()
    data_parser.get_vocab()
