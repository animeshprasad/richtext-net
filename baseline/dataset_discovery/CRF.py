## This script implements a purely feature based CRF.
import pandas as pd 
import numpy as np 
from data_reader import *
from sklearn_crfsuite import CRF 
from sklearn_crfsuite import metrics


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

##CRF takes string as labels
def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

##labels are strings
X_train = [sent2features(s) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]


crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)

crf.fit(X_train, Y_train)


labels = list(crf.classes_)
y_pred = crf.predict(X_test)
##average F1
# metrics.flat_f1_score(Y_test, y_pred,
#                       average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    Y_test, y_pred, labels=sorted_labels, digits=3
))


preds = [[int(a) for a in x] for x in y_pred]
##record the prediceted start and end index
with open('../../outputs/CRF_preds', 'w') as fout:
    with open('../../data/test.txt', 'r') as test:
        test_list = test.readlines()
        for i in range(len(preds)):
            sent = test_list[i].strip().split()
            data_id = int(sent[2])
            pub_id = int(sent[3])

            first = 0
            j = 0
            string = ''
            no_mention = True
            while j<len(preds[i]):
                while j<len(preds[i]) and preds[i][j]== 0:
                    j+=1
                if j<len(preds[i]) and preds[i][j] == 1:
                    no_mention=False
                    start = j
                    while j+1<len(preds[i]) and preds[i][j+1]==1:
                        j+=1
                    end = j 
                    if first > 0:
                        string += " | "
                    string += (str(start)+' '+str(end)+' '+str(data_id)+' '+str(pub_id))
                    j+=1
                    first += 1
            if no_mention:
                fout.write("0 0 0 "+str(pub_id)+'\n')
            else:
                fout.write(string+'\n')








