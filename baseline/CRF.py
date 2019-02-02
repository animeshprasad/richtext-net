import pandas as pd 
import numpy as np 
from data_reader import *
from evaluate_new import *
from sklearn_crfsuite import CRF 
from sklearn_crfsuite import metrics
import re
import os

## prepare data for doc evaluation
doc_dir = "../../data/all_test_docs/"
def get_doc_test(gold, text):
    ## gold: gold data
    ## text: full text file
    test_labels = []
    test_doc = []
    with open(doc_dir+gold, 'r') as doc_labels, open(doc_dir+text, 'r') as doc_text:
        d_labels = doc_labels.readlines()
        d_text = doc_text.readlines()
        assert len(d_labels) == len(d_text), "Mismatch"
        for i in range(len(d_labels)):
            ## label: start_id end_id data_id pub_id
            test_labels.append(d_labels[i].strip())
            
            text = d_text[i].strip()
            text = re.sub('\d', '0', text)
            text = re.sub('[^ ]- ', '', text)
            
            test_doc.append(text)
    return test_labels, test_doc

## convert one doc data to (text, label) format
def read_doc(doc, labels):
    doc = doc.strip().split()
    labels = labels.strip().split('|')
    labels = [la.split() for la in labels]
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            labels[i][j] = int(labels[i][j])

    res_labels = [0]*len(doc)
    for la in labels:
        if la[2]!=0:
            start = la[0]
            end = la[1]
            res_labels[start : end+1] = [1]*(end+1-start)
    return [(doc[i], str(res_labels[i])) for i in range(len(doc))]

## make prediction of one doc
## split into segments first then combine results
def doc_pred(model, doc, MAXLEN=30):
    splits = []
    for i in range(0, len(doc), MAXLEN):
        splits.append(doc[i : i+MAXLEN])
    preds = model.predict(splits)
    preds = [p for pd in preds for p in pd]  ## flatten
    return preds

## convert to crf features
def word2features(sent, i):
    word = sent[i][0]

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


## train and return a model
def run(neg_ratio=0.10, val_ratio=0.05, data_dir='../../data/data_60/'):
    train_sents, val_sents = data_sampler(neg_ratio, val_ratio, data_dir)

    train_sents = get_sents(train_sents)
    val_sents = get_sents(val_sents)

    X_train = [sent2features(s) for s in train_sents]
    Y_train = [sent2labels(s) for s in train_sents]

    X_val = [sent2features(s) for s in val_sents]
    Y_val = [sent2labels(s) for s in val_sents]

    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=False)

    crf.fit(X_train, Y_train)

    labels = list(crf.classes_)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    Y_pred = crf.predict(X_val)

    ## print validation f1 score
    print ('evaluating: neg ratio: {}, val ratio: {}'.format(neg_ratio, val_ratio))
    print (metrics.flat_classification_report(
        Y_val, Y_pred, labels=sorted_labels, digits=3
    ))

    print ("Evaluate: dev seg exact")
    p, r, f = seg_exact_match(Y_pred, Y_val, out+'seg_'+str(neg_ratio)+'neg', '../../data/val_segs/'+'seg_'+str(neg_ratio)+'neg')

    return crf

def doc_eval(model, doc_test, doc_out_dir, gold_dir, MAXLEN=60):
    '''
    model: trained model 
    doc_test: processed doc test input
    doc_out_dir: dir to store predicted results
    gold_dir: gold data dir, for evaluation
    prediction format: start_id end_id
    '''
    doc_preds = [doc_pred(model, d, MAXLEN) for d in doc_test]
    doc_preds = [[int(a) for a in x] for x in doc_preds]
    with open(doc_out_dir, 'w') as fout:
        for i in range(len(doc_preds)):
            first = 0
            j = 0
            string = ''
            no_mention = True
            while j<len(doc_preds[i]):
                while j<len(doc_preds[i]) and doc_preds[i][j]== 0:
                    j+=1
                if j<len(doc_preds[i]) and doc_preds[i][j] == 1:
                    no_mention=False
                    start = j
                    while j+1<len(doc_preds[i]) and doc_preds[i][j+1]==1:
                        j+=1
                    end = j 
                    if first > 0:
                        string += " | "
                    string += (str(start)+' '+str(end))
                    j+=1
                    first += 1
            if no_mention:
                fout.write("-1 -1"+'\n')
            else:
                fout.write(string+'\n')
    print ('evaluating data from: ', doc_out_dir)
    print ('doc exact: ', doc_exact_match(doc_out_dir, gold_dir))
    print ('doc partial: ', doc_partial_match(doc_out_dir, gold_dir))


if __name__=='__main__':
    doc_test_y, doc_test_x = get_doc_test('test_doc_gold', 'test_docs')
    doc_tests = [read_doc(doc_test_x[d], doc_test_y[d]) for d in range(len(doc_test_x))]
    doc_tests = [sent2features(s) for s in doc_tests]

    zero_shot_y, zero_shot_x = get_doc_test('zero_shot_doc_gold', 'zero_shot_docs')
    zero_shot_tests = [read_doc(zero_shot_x[d], zero_shot_y[d]) for d in range(len(zero_shot_x))]
    zero_shot_tests = [sent2features(s) for s in zero_shot_tests]

    # 0.01, 0.05, 0.10, 0.20, 0.40, 0.80
    # 0.00, 0.20, 0.40, 0.60, 0.80, 1.00
    # for i in [0.00]:
    #     MAXLEN = 20
    #     DIR = '../../data/data_20/'
    #     out = '../../outputs/'
    #     if not os.path.exists(out):
    #         os.makedirs(out)
    #     crf = run(neg_ratio=i, val_ratio=0.05, data_dir=DIR)
    #     print ("Evaluate on doc tests")
    #     doc_eval(crf, doc_tests, out+'doc_'+str(MAXLEN)+'_'+str(i)+'neg', '../../data/all_test_docs/test_doc_gold', MAXLEN)
    #     print ("Evaluate on zero shot tests")
    #     doc_eval(crf, zero_shot_tests, out+'zeroshot_'+str(MAXLEN)+'_'+str(i)+'neg', '../../data/all_test_docs/zero_shot_doc_gold', MAXLEN)

    for i in [0.0125, 0.025, 0.05, 0.10, 0.20, 0.40]:
        print ("NEG ratio: ", i)
        MAXLEN = 40
        DIR = '../../data/data_40/'
        out = '../../outputs/'
        if not os.path.exists(out):
            os.makedirs(out)
        crf = run(neg_ratio=i, val_ratio=0.05, data_dir=DIR)
        print ("Evaluate on doc tests")
        doc_eval(crf, doc_tests, out+'doc_'+str(MAXLEN)+'_'+str(i)+'neg', '../../data/all_test_docs/test_doc_gold', MAXLEN)
        print ("Evaluate on zero shot tests")
        doc_eval(crf, zero_shot_tests, out+'zeroshot_'+str(MAXLEN)+'_'+str(i)+'neg', '../../data/all_test_docs/zero_shot_doc_gold', MAXLEN)







