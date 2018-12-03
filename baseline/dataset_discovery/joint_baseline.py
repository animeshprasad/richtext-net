#This script implements a joint model
import pandas as pd 
import numpy as np 
import json
import os
from keras import layers
from keras import regularizers
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential 
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input, Flatten
from keras.callbacks import EarlyStopping
from sklearn import metrics 
from keras_contrib.layers import CRF
import keras

def read_sent(sent):
	sent = sent.strip().split()
	start = int(sent[0])
	end = int(sent[1])
	sent = sent[4:]
	labels = [0]*len(sent)
	labels[start : end+1] = [1]*(end+1-start)
	return [(sent[i], str(labels[i])) for i in range(len(sent))]

def get_sents(data_dir):
	data = open(data_dir, 'r')
	data_list = data.readlines()

	sentences = [read_sent(sent) for sent in data_list]
	labels = [int(sent.strip().split()[2]) for sent in data_list]
	data.close()
	return sentences, labels

train_sents, train_dataset_id = get_sents("../../data/train.txt")
val_sents, val_dataset_id = get_sents("../../data/validate.txt")
test_sents, test_dataset_id = get_sents("../../data/test.txt")


def sent2labels(sent):
	return [int(label) for token, label in sent]

def sent2tokens(sent):
	return [token for token, label in sent]

data_set = pd.read_json('../../../train_test/data_sets.json', encoding='utf-8')
#note: dataset_id = index + 1
data_description = data_set["description"].values

maxlen = 100
emb_dim = 50
HIDDEN_DIM = 256
EPOCHS = 10  ## train more epochs with GPU, it takes 1h per epoch on my CPU
NEG_RATIO = 3
BATCH_SIZE = 10
DATASET_CLASS = len(data_description) 
MODEL_NAME = "LSTM"


# Add a sentence for no mention case
data_description = list(data_description)
data_description.insert(0, "There is no mention.")

X_train = [sent2tokens(s) for s in train_sents]
Y_train = [sent2labels(s) for s in train_sents]
X_val = [sent2tokens(s) for s in val_sents]
Y_val = [sent2labels(s) for s in val_sents]
X_test = [sent2tokens(s) for s in test_sents]
Y_test = [sent2labels(s) for s in test_sents]


##load glove
embedding_index = {}
f = open('../glove.6B.50d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embedding_index[word] = coefs
f.close()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train+X_val+X_test)
word_index = tokenizer.word_index

##hyperparameters
vocab_size = len(word_index)+1


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

X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(X_test)


X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

Y_train = np.asarray(Y_train)
Y_val = np.asarray(Y_val)
Y_test_arr = np.asarray(Y_test)
Y_test = np.asarray(Y_test)

N = X_train.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices] 

##labels for mention detection
#labels need to be 3D
Y_train = np.expand_dims(Y_train, axis=2)
Y_val = np.expand_dims(Y_val, axis=2)
Y_test = np.expand_dims(Y_test, axis=2)

Y_train = keras.utils.to_categorical(Y_train)
Y_val = keras.utils.to_categorical(Y_val)
Y_test = keras.utils.to_categorical(Y_test)


##labels for dataset
Y2_train = np.asarray(train_dataset_id)
Y2_val = np.asarray(val_dataset_id)
Y2_test = np.asarray(test_dataset_id)
Y2_train = keras.utils.to_categorical(Y2_train, num_classes=DATASET_CLASS)
Y2_val = keras.utils.to_categorical(Y2_val, num_classes=DATASET_CLASS)
Y2_test = keras.utils.to_categorical(Y2_test, num_classes=DATASET_CLASS)


##build model
input = Input(shape=(maxlen,))
emb = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input)
[lstm_seq, state_h_fw, state_h_bw, state_c_fw, state_c_bw] = Bidirectional(LSTM(100, return_sequences=True, return_state=True, recurrent_dropout=0.2))(emb) 
state_h = layers.Concatenate(axis=-1)([state_h_fw, state_h_bw])
labels = TimeDistributed(Dense(50, activation='relu'))(lstm_seq)
##use CRF instead of Dense
crf = CRF(2)
mention = crf(labels)

data_id = Dense(DATASET_CLASS, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(state_h)
model = Model(input, [mention, data_id])

model.compile(optimizer='adam', loss=[crf.loss_function, 'categorical_crossentropy'], metrics=['accuracy']) 
history = model.fit(X_train, [Y_train, Y2_train], batch_size=64, epochs=10, 
                   validation_data=(X_val, [Y_val, Y2_val]))


[preds, data_id] = model.predict(X_test)
test = [[np.argmax(y) for y in x] for x in preds]
test_arr = np.asarray(test)
test = np.reshape(test_arr, (-1))

print (metrics.precision_recall_fscore_support(np.reshape(Y_test_arr,(-1)), test, average=None,
                                              labels=[0, 1]))


ids = [np.argmax(a) for a in data_id]
preds = test_arr
##record the prediceted start and end index
with open('../../outputs/joint_baseline_preds', 'w') as fout:
	with open('../../data/test.txt', 'r') as test:
		test_list = test.readlines()
		for i in range(len(preds)):
			sent = test_list[i].strip().split()
			data_id = ids[i]
			pub_id = int(sent[3])

			##no mention
			if data_id == 0:
				fout.write("0 0 0 "+str(pub_id)+'\n')
				continue 

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








