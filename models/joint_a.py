import pandas as pd 
import numpy as np 
from data_reader import *
from evaluate_new import *
from nltk.tokenize import SpaceTokenizer
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, Sequential
from keras import regularizers
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, Flatten
from sklearn import metrics
import re
import os
from keras_contrib.layers import CRF
import keras
from keras import layers
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import ModelCheckpoint, EarlyStopping

def data_loader(f):
    sents = []
    labels = []
    for line in f:
        line = line.strip().split()
        labels.append(int(line[2]))
        sents.append(' '.join(line[4:]))
    return sents, labels


def prep_data(neg_ratio=0.0125, val_ratio=0.05, data_dir='../../data/data_40/', maxlen=40, emb_dim=300):
    train_list, val_list = data_sampler(neg_ratio, val_ratio, data_dir)

    train_sents = get_sents(train_list)
    val_sents = get_sents(val_list)
    
    _, train_labels = data_loader(train_list)
    _, val_labels = data_loader(val_list)

    X_train = [sent2tokens(s) for s in train_sents]
    Y_train_seq = [sent2labels(s) for s in train_sents]

    X_val = [sent2tokens(s) for s in val_sents]
    Y_val_seq = [sent2labels(s) for s in val_sents]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index

    vocab_size = len(word_index)+1
    print ("Vocab size: ", vocab_size)

    all_embs = np.stack(embedding_index.values())
    emb_mean, emb_std = np.mean(all_embs), np.std(all_embs)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, emb_dim))
    counter = 0
    # embedding_matrix = np.zeros((vocab_size, emb_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            counter += 1
        else:
            embedding_matrix[i] = np.random.randn(emb_dim)
    print ("{}/{} words covered in glove".format(counter, vocab_size))

    X_train = tokenizer.texts_to_sequences(X_train)
    X_val = tokenizer.texts_to_sequences(X_val)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)

    Y_train_seq = np.asarray(Y_train_seq)
    Y_val_seq = np.asarray(Y_val_seq)
    
    #labels need to be 3D
    Y_train_seq = np.expand_dims(Y_train_seq, axis=2)
    Y_val_seq = np.expand_dims(Y_val_seq, axis=2)
    
    Y_train_cls = np.asarray(train_labels)
    Y_val_cls = np.asarray(val_labels)
    
    Y_train_cls = keras.utils.to_categorical(Y_train_cls, num_classes=DATASET_CLASS)
    Y_val_cls = keras.utils.to_categorical(Y_val_cls, num_classes=DATASET_CLASS)

    return X_train, Y_train_seq, Y_train_cls, X_val, Y_val_seq, Y_val_cls, embedding_matrix, vocab_size, tokenizer


## load test data for classification task
def load_test(dir='../../data/all_test_docs/'):
    test_labels = []
    test_sents = []
    zeroshot_labels = []
    zeroshot_sents = []
    with open(dir+'test_doc_gold', 'r') as f:
        fl = f.readlines()
        for line in fl:
            labels = []
            line = line.strip().split('|')
            for data in line:
                labels.append(int(data.split()[2]))
            test_labels.append(labels)
    with open(dir+'test_docs', 'r') as f:
        test_sents = f.readlines()
        
    with open(dir+'zero_shot_doc_gold', 'r') as f:
        fl = f.readlines()
        for line in fl:
            labels = []
            line = line.strip().split('|')
            for data in line:
                labels.append(int(data.split()[2]))
            zeroshot_labels.append(labels)
    with open(dir+'zero_shot_docs', 'r') as f:
        zeroshot_sents = f.readlines()
        
    return test_labels, test_sents, zeroshot_labels, zeroshot_sents


## prepare data for doc evaluation
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


def sent2labels(sent):
    return [int(label) for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


def run(X_train, Y_train_seq, Y_train_cls, X_val, Y_val_seq, Y_val_cls, embedding_matrix, vocab_size, maxlen=40, emb_dim=300, neg_ratio=0.0125, hidden_dim=100, drop=0.2, r_drop=0.1, epochs=10):
    ##build model
    input = Input(shape=(maxlen,))
    emb = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input)
    [lstm_seq, state_h_fw, state_h_bw, state_c_fw, state_c_bw] = Bidirectional(LSTM(hidden_dim, return_sequences=True, return_state=True, recurrent_dropout=r_drop))(emb) 
    [lstm_seq, state_h_fw, state_h_bw, state_c_fw, state_c_bw] = Bidirectional(LSTM(hidden_dim, return_sequences=True, return_state=True, recurrent_dropout=r_drop))(lstm_seq) 
    state_h = layers.Concatenate(axis=-1)([state_h_fw, state_h_bw])
    labels = TimeDistributed(Dense(hidden_dim//2, activation='relu'))(lstm_seq)
    lstm_seq_drop = TimeDistributed(Dropout(drop))(labels)
    crf = CRF(2)
    mention = crf(labels)
   
    state_h = Dropout(drop)(state_h)
    data_id = Dense(DATASET_CLASS, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(state_h)
    model = Model(input, [mention, data_id])
    
    Y_train_seq_2 = keras.utils.to_categorical(Y_train_seq)
    Y_val_seq_2 = keras.utils.to_categorical(Y_val_seq)

    model.compile(optimizer='adam', loss=[crf.loss_function, 'categorical_crossentropy'], metrics=['accuracy']) 
#     history = model.fit(X_train, [Y_train_seq, Y_train_cls], batch_size=64, epochs=10, validation_data=(X_val, [Y_val_seq, Y_val_cls]))
    history = model.fit(X_train, [Y_train_seq_2, Y_train_cls], batch_size=64, epochs=epochs, validation_data=(X_val, [Y_val_seq_2, Y_val_cls]))

    
    [preds, data_ids] = model.predict(X_val)
    
    test = [[np.argmax(y) for y in x] for x in preds]
    test_arr = np.asarray(test)
    test_arr = np.reshape(test_arr, (-1))
    
    print ("Dev Sequence Labeling:")
    print (metrics.precision_recall_fscore_support(np.reshape(Y_val_seq,(-1)), test_arr, average=None,
                                              labels=[0, 1]))

    Y_val_ = np.squeeze(Y_val_seq)

    print ("Evaluate: dev seg exact")
    pred_out_dir = out_dir+'seg_'+str(neg_ratio)+'neg'
    gold_dir = '../../data/val_segs/'+'seg_'+str(neg_ratio)+'neg'
    p, r, f = seg_exact_match(test, Y_val_, pred_out_dir, gold_dir)
    
    print ("Evaluate: dataset class score:")
    preds = [[np.argmax(y)] for y in data_ids]
    Y_v = [[np.argmax(y)] for y in Y_val_cls]
    p, r, f = classify_score(preds, Y_v)
    
    return model, history


def doc_pred(model, doc, tokenizer, MAXLEN=40):
    splits = []
    for i in range(0, len(doc), MAXLEN):
        splits.append(doc[i: i+MAXLEN])
    splits = tokenizer.texts_to_sequences(splits)
    splits = pad_sequences(splits, maxlen=MAXLEN)
    [preds, data_ids] = model.predict(splits)
    seq_preds = [np.argmax(y) for x in preds for y in x]
    cls_preds = [np.argmax(y) for y in data_ids]
    return seq_preds, cls_preds


def doc_eval_seq(model, tokenizer, doc_test, doc_out_dir, gold_dir, MAXLEN=40):
    doc_preds = []
    for d in doc_test:
        seq_preds, cls_preds = doc_pred(model, d, tokenizer, MAXLEN)
        doc_preds.append(seq_preds)
#     doc_preds = [doc_pred(model, d, tokenizer, MAXLEN) for d in doc_test]
    
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



def doc_eval_cls(model, doc, labels, tokenizer, MAXLEN=40):
    preds = []
    for d in doc:
        seq_preds, cls_preds = doc_pred(model, d, tokenizer, MAXLEN)
        preds.append(cls_preds)
        
#     preds = [doc_pred(model, d, tokenizer) for d in doc]
    p, r, f = classify_score(preds, labels)
    return p, r, f



def doc_eval(model, tokenizer, doc_test, cls_labels, doc_out_dir, gold_dir, MAXLEN=40):
    doc_preds_seq = []
    doc_preds_cls = []
    for d in doc_test:
        seq_preds, cls_preds = doc_pred(model, d, tokenizer, MAXLEN)
        doc_preds_seq.append(seq_preds)
        doc_preds_cls.append(cls_preds)
    
    doc_preds = doc_preds_seq
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
    print ('Evaluate seq:')
    print ('evaluating data from: ', doc_out_dir)
    print ('doc exact: ', doc_exact_match(doc_out_dir, gold_dir))
    print ('doc partial: ', doc_partial_match(doc_out_dir, gold_dir))
    
    print ('Evaluate cls:')
    p, r, f = classify_score(doc_preds_cls, cls_labels)
    return p, r, f


##load glove
embedding_index = {}
f = open('../../glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()


# Load all test data for both tasks
threshold = 0.5
doc_dir = "../../data/all_test_docs/"
doc_test_y, doc_test_x = get_doc_test('test_doc_gold', 'test_docs')
doc_tests = [read_doc(doc_test_x[d], doc_test_y[d]) for d in range(len(doc_test_x))]
doc_tests = [sent2tokens(s) for s in doc_tests]

zero_shot_y, zero_shot_x = get_doc_test('zero_shot_doc_gold', 'zero_shot_docs')
zero_shot_tests = [read_doc(zero_shot_x[d], zero_shot_y[d]) for d in range(len(zero_shot_x))]
zero_shot_tests = [sent2tokens(s) for s in zero_shot_tests]

MAXLEN = 40
DIR = '../../data/data_40/'
out_dir = '../../outputs/'

if not os.path.exists(out_dir):
        os.makedirs(out_dir)

test_labels, test_sents, zeroshot_labels, zeroshot_sents = load_test()
DATASET_CLASS = 10349  ## one class for no mention (#0)


# Get Training Data
neg_ratio = 0.0125
X_train, Y_train_seq, Y_train_cls, X_val, Y_val_seq, Y_val_cls, embedding_matrix, vocab_size, tokenizer = prep_data(neg_ratio=neg_ratio)


hidden_dim = 100
drop = 0.1
r_drop=0.1
model, history = run(X_train, Y_train_seq, Y_train_cls, X_val, Y_val_seq, Y_val_cls, embedding_matrix, vocab_size, neg_ratio=neg_ratio, hidden_dim=hidden_dim, drop=drop, r_drop=r_drop)


hidden_dim = 300
epochs=20
drop = 0.3
r_drop=0.3
model, history = run(X_train, Y_train_seq, Y_train_cls, X_val, Y_val_seq, Y_val_cls, embedding_matrix, vocab_size, neg_ratio=neg_ratio, hidden_dim=hidden_dim, drop=drop, r_drop=r_drop, epochs=epochs)
print ('Test Set:')
doc_eval(model, tokenizer, doc_tests, test_labels, out_dir+'doc_40_'+str(neg_ratio)+'neg', '../../data/all_test_docs/test_doc_gold')
print ('Zero-shot Test:')
doc_eval(model, tokenizer, zero_shot_tests, zeroshot_labels, out_dir+'zeroshot_40_'+str(neg_ratio)+'neg', '../../data/all_test_docs/zero_shot_doc_gold')













