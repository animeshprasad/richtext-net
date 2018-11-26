import keras 
from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 

sentences = open("sentences", 'r')  
f_labels = open("labels", 'r')

X = [sent.strip() for sent in sentences.readlines()]
Y = [int(num.strip()) for num in f_labels.readlines()]

maxlen = 200
train_samples = 0.8
val_samples = 0.05
test_samples = 0.15
vocab_size = 100000

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)

word_index = tokenizer.word_index
print ("Found %s unique tokens."%len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
lables = np.asarray(Y)
total_class = max(labels)  # total number of classes
labels = keras.utils.to_categorical(labels)

N = data.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[: int(N*train_samples)]
y_train = labels[: int(N*train_samples)]
x_val = data[int(N*train_samples) : int(N*(train_samples+val_samples))]
y_val = labels[int(N*train_samples) : int(N*(train_samples+val_samples))]
x_test = data[int(N*(train_samples+val_samples)) : ]
y_test = labels[int(N*(train_samples+val_samples)) : ]


#not using glove embeddings
model = Sequential()
model.add(layers.Embedding(vocab_size, 40))
model.add(layers.Bidirectional(layers.LSTM(64, dropout=0.2, recurrent_dropout=0.4)))
model.add(layers.Dense(total_class+1, activation="softmax"))

callbacks_list = [
	keras.callbacks.EarlyStopping(
		monitor='val_acc',
		patience=2),
	keras.callbacks.ModelCheckpoint(
		filepath="LSTM/LSTMbaseline.{epoch:02d}-{val_loss:.2f}.h5",
		save_best_only=True)
]

model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['acc'])

model.fit(x_train, y_train,
		  epochs=15,
		  batch_size=128,
		  callbacks=callbacks_list,
		  validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)









