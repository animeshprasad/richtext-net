import keras 
from keras import layers
from keras.layers import Embedding, Dropout, Dense, LSTM, Bidirectional, Input, Dense, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np 
from tqdm import tqdm_notebook as tqdm
import json
import pandas as pd 
import os
from data_reader import *
from evaluate_new import *

def data_loader(f):
    sents = []
    labels = []
    for line in f:
        line = line.strip().split()
        labels.append(int(line[2]))
        sents.append(' '.join(line[4:]))
    return sents, labels


##load glove
embedding_index = {}
f = open('../../glove.6B.300d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()


def prep_data(neg_ratio=0.0125, val_ratio=0.05, data_dir='../../data/data_40/', maxlen=40, emb_dim=300):
    train_list, val_list = data_sampler(neg_ratio, val_ratio, data_dir)

    train_sents, train_labels = data_loader(train_list)
    val_sents, val_labels = data_loader(val_list)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sents)
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

    X_train = tokenizer.texts_to_sequences(train_sents)
    X_val = tokenizer.texts_to_sequences(val_sents)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)

    Y_train = np.asarray(train_labels)
    Y_val = np.asarray(val_labels)
    
    Y_train = keras.utils.to_categorical(Y_train, num_classes=DATASET_CLASS)
    Y_val = keras.utils.to_categorical(Y_val, num_classes=DATASET_CLASS)

    return X_train, Y_train, X_val, Y_val, embedding_matrix, vocab_size, tokenizer


## load test data
## test sents are complete docs, we need to identify all the dataset classes
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


def run(X_train, Y_train, X_val, Y_val, embedding_matrix, vocab_size, maxlen=40, emb_dim=300, hidden_dim=200, drop=0.1, r_drop=0.1):
    ##build model
    input = Input(shape=(maxlen,))
    model = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input)
    model = Dropout(drop)(model)
    model = Bidirectional(LSTM(hidden_dim, recurrent_dropout=r_drop))(model)
    model = Dropout(drop)(model)
    out = Dense(DATASET_CLASS, activation='sigmoid')(model)

    model = Model(input, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlyStop = [EarlyStopping(monitor='val_acc', patience=2)]
    history = model.fit(X_train, Y_train, batch_size=64, epochs=20, validation_data=(X_val, Y_val), 
        callbacks=earlyStop) 
        
    preds = model.predict(X_val)
    preds = [[np.argmax(y)] for y in preds]
    Y_v = [[np.argmax(y)] for y in Y_val]
    p, r, f = classify_score(preds, Y_v)
    
    return model, history, p, r, f


def doc_pred(model, doc, tokenizer, MAXLEN=40):
    splits = []
    for i in range(0, len(doc), MAXLEN):
        splits.append(doc[i : i+MAXLEN])
    splits = tokenizer.texts_to_sequences(splits)
    splits = pad_sequences(splits, maxlen=MAXLEN)
    preds = model.predict(splits)
    preds = [np.argmax(y) for y in preds]
    return preds


def doc_eval(model, doc, labels, tokenizer, MAXLEN=40):
    preds = [doc_pred(model, d, tokenizer) for d in doc]
    p, r, f = classify_score(preds, labels)
    return p, r, f

if __name__=='__main__':
	test_labels, test_sents, zeroshot_labels, zeroshot_sents = load_test()
	DATASET_CLASS = 10349  ## one class for no mention (#0)
	X_train, Y_train, X_val, Y_val, embedding_matrix, vocab_size, tokenizer = prep_data(neg_ratio=0.0125)
	model, history, p, r, f  = run(X_train, Y_train, X_val, Y_val, embedding_matrix, vocab_size, hidden_dim=128, drop=0, r_drop=0)
	doc_eval(model, test_sents, test_labels, tokenizer)
	doc_eval(model, zeroshot_sents, zeroshot_labels, tokenizer)






