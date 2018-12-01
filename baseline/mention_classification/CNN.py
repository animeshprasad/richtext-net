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

def data_loader(file):
	with open(file, 'r') as f:
		sents = []
		labels = []
		for line in f:
			line = line.strip().split()
			labels.append(int(line[2]))
			sents.append(' '.join(line[4:]))
	return sents, labels


train_sents, train_labels = data_loader("../../data/train.txt")
val_sents, val_labels = data_loader("../../data/validate.txt")
test_sents, test_labels = data_loader("../../data/test.txt")


data_set = pd.read_json('../../../train_test/data_sets.json', encoding='utf-8')
#note: dataset_id = index + 1
data_description = data_set["description"].values


# Add a sentence for no mention case
data_description = list(data_description)
data_description.insert(0, "There is no mention.")


maxlen = 100
#vocab_size = 40000 ##more than 80K unique tokens
emb_dim = 50
HIDDEN_DIM = 256
EPOCHS = 10  ## train more epochs with GPU, it takes 1h per epoch on my CPU
NEG_RATIO = 3
BATCH_SIZE = 10
DATASET_CLASS = len(data_description) 
MODEL_NAME = "CNN"

#actual batch size = BATCH_SIZE * (1 + NEG_RATIO)

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
tokenizer.fit_on_texts(train_sents+val_sents+test_sents)
X_train = tokenizer.texts_to_sequences(train_sents)
X_val = tokenizer.texts_to_sequences(val_sents)
X_test = tokenizer.texts_to_sequences(test_sents)

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



X_train = pad_sequences(X_train, maxlen=maxlen)
X_val = pad_sequences(X_val, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
Y_train = np.asarray(train_labels)
Y_val = np.asarray(val_labels)
Y_test = np.asarray(test_labels)
Y_train = keras.utils.to_categorical(Y_train, num_classes=DATASET_CLASS)
Y_val = keras.utils.to_categorical(Y_val, num_classes=DATASET_CLASS)
Y_test = keras.utils.to_categorical(Y_test, num_classes=DATASET_CLASS)


##randomly shuffle data and labels
##np.random.seed(0)
N = X_train.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices] 


def build_model():
	#CNN model, 1D conv
	model = Sequential()
	model.add(layers.Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
	model.add(layers.Conv1D(32, 7, activation='relu'))
	model.add(layers.MaxPooling1D(5))
	model.add(layers.Conv1D(32, 7, activation='relu'))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(DATASET_CLASS, activation="sigmoid"))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model 

model = build_model()

callbacks_list = [
	keras.callbacks.EarlyStopping(
		monitor='val_acc',
		patience=2)
]

history = model.fit(X_train, Y_train, 
					epochs=EPOCHS,
					batch_size=128,
					validation_data=(X_val, Y_val),
					callbacks=callbacks_list)

import matplotlib.pyplot as plt
%matplotlib inline

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Trianing and validation loss')
plt.legend()

plt.show()

print ('CNN model test accuracy: ', model.evaluate(X_test, Y_test))











