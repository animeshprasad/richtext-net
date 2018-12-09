#!/usr/bin/env python
# coding: utf-8

# In[62]:


import keras
import re
from keras import layers
from keras import regularizers
from keras.layers import TimeDistributed, Embedding, Dropout, Dense, LSTM, Bidirectional, Input, Dense, Flatten, Conv1D, GlobalMaxPooling1D, Permute, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk import WordPunctTokenizer
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K 
import numpy as np 
from tqdm import tqdm_notebook as tqdm
import json
import pandas as pd 
import os
import random

def data_loader(file):
    with open(file, 'r') as f:
        sents = []
        labels = []
        for line in f:
            line = line.strip().split()
            labels.append(int(line[2]))
            sents.append(' '.join(line[4:]))
    return sents, labels


# In[63]:


train_sents, train_labels = data_loader("../../data/train.txt")
val_sents, val_labels = data_loader("../../data/validate.txt")
test_sents, test_labels = data_loader("../../data/test.txt")

# # Add a sentence for no mention case
# data_description = list(data_description)
# data_description.insert(0, "There is no mention.")


# In[64]:


maxlen = 30
mention_len = 50
emb_dim = 50
HIDDEN_DIM = 64
EPOCHS = 10  
NEG_RATIO = 3
BATCH_SIZE = 10
MODEL_NAME = "LSTM"
DATASET_CLASS = max(train_labels+val_labels+test_labels)+1


# In[65]:


##only first 1330 datasets have mentions


# In[66]:


datasets = pd.read_json('../../train_test/data_sets.json', encoding='utf-8')
dataset_mention = datasets["mention_list"].values


# In[67]:


mentions = dataset_mention[: DATASET_CLASS-1]


# In[68]:


def choose_longest(mention):
    if len(mention)==0:
        return "UNK"
    else:
        mention = [ " ".join(list(WordPunctTokenizer().tokenize(re.sub('[^ ]- ', '', item)))) for item in mention]
        idx = np.argmax(np.asarray([len(a.strip().split()) for a in mention]))
        return mention[idx]


# In[69]:


longest_mentions = [choose_longest(m) for m in mentions]
longest_mentions = ["UNK"] + longest_mentions


# In[70]:


len(longest_mentions)


# In[71]:


##load glove
embedding_index = {}
f = open('../../glove/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()

###NOT using dataset info anymore
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sents+val_sents+test_sents+longest_mentions)
X_train = tokenizer.texts_to_sequences(train_sents)
X_val = tokenizer.texts_to_sequences(val_sents)
X_test = tokenizer.texts_to_sequences(test_sents)
long_mentions = tokenizer.texts_to_sequences(longest_mentions)

word_index = tokenizer.word_index
vocab_size = len(word_index)+1
print ("vocab size: ", vocab_size)

embedding_matrix = np.zeros((vocab_size, emb_dim))
counter = 0
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        counter += 1
    else:
        embedding_matrix[i] = np.random.randn(emb_dim)
print ("{}/{} words covered in glove".format(counter, vocab_size))


# In[72]:


max([len(m) for m in long_mentions])
## mention_len=10


# In[73]:


X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
Y_train = np.asarray(train_labels)
Y_val = np.asarray(val_labels)
Y_test = np.asarray(test_labels)
# Y_train = keras.utils.to_categorical(Y_train, num_classes=DATASET_CLASS)
# Y_val = keras.utils.to_categorical(Y_val, num_classes=DATASET_CLASS)
# Y_test = keras.utils.to_categorical(Y_test, num_classes=DATASET_CLASS)

long_men = pad_sequences(long_mentions, maxlen=mention_len)


# X_Train = []
# X_Val =  []
# X_Test = []
# Y_Train =  []
# Y_Val =  []
# Y_Test =  []
LM_train = []
LM_val = []
LM_test = []


def zeros_shot_data(X_train, Y_train, LM_train, neg_rate=4):
    X_Train = []
    Y_Train =  []
    LM_Train = []
    index = []
    for i in range(len(X_train)):
        X_Train.append(X_train[i])
        Y_Train.append([0, 1])
        LM_Train.append(long_men[Y_train[i]])
        index.append(Y_train[i])
        if neg_rate == -1:
            for j in range(long_men.shape[0]):
                if j != Y_train[i]:
                    X_Train.append(X_train[i])
                    Y_Train.append([1, 0])
                    LM_Train.append(long_men[j])
                    index.append(j)
        else:
            for j in range(neg_rate):
                k = random.randint(0, long_men.shape[0]-1)
                if k != Y_train[i]:
                    LM_Train.append(long_men[k])
                    X_Train.append(X_train[i])
                    Y_Train.append([1, 0])
                    index.append(k)
    return np.array(X_Train), np.array(Y_Train), np.array(LM_Train), np.array(index)


X_train, Y_train, LM_train,_ = zeros_shot_data( X_train, Y_train, LM_train, neg_rate=100)
X_val, Y_val, LM_val,_ = zeros_shot_data(X_val, Y_val, LM_val, neg_rate=100)
X_test, Y_test, LM_test, index = zeros_shot_data(X_test, Y_test, LM_test, neg_rate=-1)
print ('here')

print X_train.shape, Y_train.shape, LM_train.shape, X_val.shape, Y_val.shape, LM_val.shape, X_test.shape, Y_test.shape, LM_test.shape
Y_test = np.argmax(Y_test, axis =1)

# In[74]:


##randomly shuffle data and labels
##np.random.seed(0)
N = X_train.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
X_train = X_train[indices]
LM_train = LM_train[indices]
Y_train = Y_train[indices] 


# In[75]:


long_men.shape


# In[76]:


def build_model3():
    embedding_layer = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    article_input = Input(shape=(maxlen,), dtype='int32')
    article_emb = embedding_layer(article_input)

    embedding_layer2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=mention_len,
                                 trainable=False)
    dataset_input = Input(shape=(mention_len,), dtype='int32')
    dataset_emb = embedding_layer2(dataset_input)

    #article_lstm = LSTM(HIDDEN_DIM)
    #article_vector = article_lstm(article_emb)

    conv = Conv1D(emb_dim, 10)

    article_vector = conv(article_emb)
    article_vector = GlobalMaxPooling1D()(article_vector)
    article_vector = Dense(HIDDEN_DIM)(article_vector)


    dataset_vector = conv(dataset_emb)
    dataset_vector = GlobalMaxPooling1D()(dataset_vector)
    dataset_vector = Dense(HIDDEN_DIM)(dataset_vector)
    #dataset_vector = article_lstm(dataset_emb)


    merged = layers.merge.dot([article_vector, dataset_vector], axes=1)
    # shape: (batch_size, 1)
    output = Dense(2, activation='softmax')(merged)
    #
    model = Model([article_input, dataset_input], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit([X_train, LM_train], Y_train, batch_size=64, epochs=2,
                        validation_data=([X_val, LM_val], Y_val), class_weight=None) #{0:0.25, 1:0.75})

    k = model.predict([X_test, LM_test])
    k = np.argmax(k, axis=1)

    print np.bincount(k)

    from sklearn.metrics import precision_recall_fscore_support
    print precision_recall_fscore_support(Y_test, k)


    return model




def build_model2():
    embedding_layer = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    article_input = Input(shape=(maxlen,), dtype='int32')
    article_emb = embedding_layer(article_input)
    article_lstm = LSTM(HIDDEN_DIM, dropout=0.2, recurrent_dropout=0.3)
    article_vector = article_lstm(article_emb)

    embedding_layer2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=mention_len,
                                 trainable=False)
    dataset_lstm = LSTM(HIDDEN_DIM, dropout=0.2, recurrent_dropout=0.3)

    mentions = []
    emb = []
    dataset_vector=[]
    for i in range(long_men.shape[0]):
        mentions.append(Input(batch_shape=(mention_len, ), dtype='int32'))
        emb.append(embedding_layer2(mentions[-1]))
        dataset_vector = dataset_lstm(emb[-1])

    #
    # preds = Lambda(dot)([article_vector, mention_vec])
    # ##shape: (batch_size, DATASET_CLASS)

    output = Dense(DATASET_CLASS, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(article_lstm)
    ##Just to add a layer of sigmoid

    model = Model( long_men, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_model():
    embedding_layer = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    article_input = Input(shape=(maxlen,), dtype='int32')
    article_emb = embedding_layer(article_input)

    
    article_lstm = LSTM(HIDDEN_DIM, dropout=0.2, recurrent_dropout=0.3)
    article_vector = article_lstm(article_emb)
    #vector shape: (batch_size, hidden_dim)
    
    ##mention input: (DATASET_CLASS, mention_len)
    
    
    mentions = Input(batch_shape=long_men.shape, dtype='int32')
    embedding_layer2 = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=mention_len, trainable=False)

    #mentions = Lambda(lambda x : K.constant(x))(long_menv)
    
    
    mention_emb = TimeDistributed(embedding_layer2)(mentions)
    ##shape: (DATASET_CLASS, mention_len, emb_dim)
    
    mention_vec = Conv1D(emb_dim, 5)(mention_emb)
    mention_vec = GlobalMaxPooling1D()(mention_vec)
    ##shape: (DATASET_CLASS, emb_dim)
    
    article_vector = Dense(emb_dim, kernel_regularizer=regularizers.l2(0.01))(article_vector)
    ##shape: (batch_size, emb_dim)
    
    mention_vec = Lambda(lambda x: K.permute_dimensions(x, (1,0)))(mention_vec)
    ##shape: (emb_dim, DATASET_CLASS)
    
    def dot(inp):
        x = inp[0]
        y = inp[1]
        return K.dot(x, y)
    
    preds = Lambda(dot)([article_vector, mention_vec])
    ##shape: (batch_size, DATASET_CLASS)
    
    output = Dense(DATASET_CLASS, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(preds)
    ##Just to add a layer of sigmoid
    
    model = Model([article_input, long_men], output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model 


# In[77]:


model = build_model3()


# In[ ]:




