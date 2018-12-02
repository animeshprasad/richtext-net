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
import json


reload(sys)
sys.setdefaultencoding('utf8')

MAX_LENGTH=30

class OutParser:

    def __init__(self, indir=None, trainindir=None, outdir=None):
        self.data_set_citations = pd.read_json('data_set_citations.json', encoding='utf-8')
        self.data_set = pd.read_json('data_sets.json', encoding='utf-8')

        self.data_set_citations.drop(self.data_set_citations.index, inplace=True)
        #self.data_set.drop(self.data_set.index, inplace=True)


        self.publications = pd.read_json(indir+'publications.json', encoding='utf-8')

        self.publications.insert(6, 'mention_list', pd.Series)
        self.publications.insert(7, 'score', pd.Series)

        mentions_per_dataset = {}
        mention_lists_perdoc = {}

        with open (outdir+'BiLSTM_CRF_preds', 'r') as pred, open(trainindir+'golden_test', 'r') as feat:
            try:
                X = feat.readlines()
                Y = pred.readlines()
                for i in range(len(X)):

                    x = X[i]
                    y = Y[i]
                    # print x
                    # print y
                    multi_predicts = y.split('|')
                    single_true = x.split()
                    for i in range(len(multi_predicts)):
                        predicts = [int(t) for t in multi_predicts[i].split()[:4]]
                        trues = single_true[4:]
                        if predicts[0] != 0 and predicts[1] != 0:
                            if predicts[3] in mention_lists_perdoc.keys():
                                mention_lists_perdoc[predicts[3]].append(' '.join(trues[predicts[0]:predicts[1]+1]))
                            else:
                                mention_lists_perdoc[predicts[3]] = [' '.join(trues[predicts[0]:predicts[1]+1])]


            except:
                    pass


        mention_lists_perdoc_unique = {}
        mention_list = []
        dataset_list = []
        for keys, values in mention_lists_perdoc.iteritems():
            total = len(values)
            values = Counter(values).most_common(len(values)/5)
            v={}
            for t1, t2  in values:
                v[t1] = t2

            for imention in v.keys():
                a={}
                a["publication_id"] =keys
                a["mention"] = imention
                a["score"] = v[imention]*1.0/total
                mention_list.append(a)

            data_to_mention={}
            data_to_mention_score = {}
            for index, row in self.data_set.iterrows():
                if type(row["mention_list"])!= type([]):
                    row["mention_list"] = [row["mention_list"]]
                    print row["mention_list"]
                    input()
                for imention in v.keys():
                    if imention in row["mention_list"]:
                        if index in data_to_mention.keys():
                            data_to_mention[index].append(imention)
                            data_to_mention_score[index] += 1
                        else:
                            data_to_mention[index] = [imention]
                            data_to_mention_score[index] = 1
                    # else:
                    #     s = [len(set(imention.split()).intersection(c)) for c in set(' '.join(row["mention_list"]).split())]
                    #     if max(s)>0.3:
                    #         if index in data_to_mention.keys():
                    #             data_to_mention[index].append(imention)
                    #             data_to_mention_score[index] += max(s)
                    #         else:
                    #             data_to_mention[index] = [imention]
                    #             data_to_mention_score[index] = max(s)


            for data_ids in data_to_mention.keys():
                b={}
                b["publication_id"] = keys
                b["data_set_id"] = self.data_set.loc[data_ids, "data_set_id"]
                b["score"] = data_to_mention_score[data_ids]/len(data_to_mention[data_ids])
                b["mention_list"] = data_to_mention[data_ids]
                dataset_list.append(b)


        with open('../data_submit/output/data_set_mentions.json ', 'w') as fout1:
            json.dump(mention_list, fout1)

        with open('../data_submit/output/data_set_citations.json ', 'w') as fout2:
            json.dump(dataset_list, fout2)











if __name__ == '__main__':
    data_parser = OutParser(indir='../data_submit/input/', trainindir='../data/', outdir='../output/')

