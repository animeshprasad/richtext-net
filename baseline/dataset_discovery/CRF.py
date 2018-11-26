import pandas as pd 
import numpy as np 
from data_reader import *

train_sents = get_sents("../../data/train.txt")
val_sents = get_sents("../../data/validate.txt")
test_sents = get_sents("../../data/test.txt")


def word2features(sent, i):
    word = sent[i][0]

    ##youmay add more features
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }

    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })

    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(s) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]

from sklearn_crfsuite import CRF 

crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

crf.fit(X_train, Y_train)

from sklearn_crfsuite import metrics

labels = list(crf.classes_)
y_pred = crf.predict(X_test)
##average F1
metrics.flat_f1_score(Y_test, y_pred,
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    Y_test, y_pred, labels=sorted_labels, digits=3
))



















