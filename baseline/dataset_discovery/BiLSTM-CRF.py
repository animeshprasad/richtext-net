#This script implements a BiLSTM model
import pandas as pd 
import numpy as np 
from data_reader import *
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn import metrics 
from keras_contrib.layers import CRF
import keras

train_sents = get_sents("../../data/train.txt")
val_sents = get_sents("../../data/validate.txt")
test_sents = get_sents("../../data/test.txt")


def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
	return [label for token, label in sent]

def sent2tokens(sent):
	return [token for token, label in sent]

X_train = [sent2tokens(s) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]
X_val = [sent2tokens(s) for s in val_sents]
Y_val = [sent2labels(s) for s in val_sents]
X_test = [sent2tokens(s) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
vocab_size = len(word_index)+1

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)

maxlen = 100

X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

Y_train = np.asarray(Y_train)
Y_val = np.asarray(Y_val)
Y_test = np.asarray(Y_test)

#labels need to be 3D
Y_train = np.expand_dims(Y_train, axis=2)
Y_val = np.expand_dims(Y_val, axis=2)
Y_test = np.expand_dims(Y_test, axis=2)

##build model
emb_dim = 50
input = Input(shape=(maxlen,))
model = Embedding(vocab_size, emb_dim, input_length=maxlen)(input)
model = Bidirectional(LSTM(100, return_sequences=True, recurrent_dropout=0.2))(model)    
model = TimeDistributed(Dense(50, activation='relu'))(model)
##use CRF instead of Dense
crf = CRF(2)
out = crf(model)

model = Model(input, out)


Y_train_2 = keras.utils.to_categorical(Y_train)
Y_val_2 = keras.utils.to_categorical(Y_val)
Y_test_2 = keras.utils.to_categorical(Y_test)

model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy]) 
earlyStop = [EarlyStopping(monitor='val_loss', patience=1)]
history = model.fit(X_train, Y_train_2, batch_size=64, epochs=10, 
                   validation_data=(X_val, Y_val_2), callbacks=earlyStop)


preds = model.predict(X_test)
test = [[np.argmax(y) for y in x] for x in preds]
test_arr = np.asarray(test)
test = np.reshape(test_arr, (-1))

print (metrics.precision_recall_fscore_support(np.reshape(Y_test,(-1)), test, average=None,
                                              labels=[0, 1]))


preds = test_arr
##record the prediceted start and end index
with open('../../outputs/BiLSTM_preds', 'w') as fout:
	with open('../../data/test.txt', 'r') as test:
		test_list = test.readlines()
		for i in range(len(preds)):
			sent = test_list[i].strip().split()
			data_id = int(sent[2])
			pub_id = int(sent[3])

			j = 0
			while j<len(preds[i]):
				while j<len(preds[i]) and preds[i][j]== 0:
					j+=1
				if j<len(preds[i]) and preds[i][j] == 1:
					start = j
					while j+1<len(preds[i]) and preds[i][j+1]==1:
						j+=1
					end = j 
					fout.write(str(start)+' '+str(end)+' '+str(data_id)+' '+str(pub_id)+'\n')
					j+=1








