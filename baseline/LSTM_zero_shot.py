import keras 
from keras import layers
from keras.layers import Embedding, Dense, Bidirectional, Input, Dense, TimeDistributed, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
import json
import pandas as pd 

data_set = pd.read_json('../../train_test/data_sets.json', encoding='utf-8')
#note: dataset_id = index + 1
data_description = data_set["description"].values

DIR = '../../train_test/golden_data'

X = []
Y = []

with open(DIR, 'r') as f:
	for line in f:
		line = line.strip()
		Y.append(int(line[0]))
		X.append(line[2:])

print (len(X), 'sampled loaded')

maxlen = 200
train_samples = 0.8
val_samples = 0.05
test_samples = 0.15
vocab_size = 100000
EMB_DIM = 50
HIDDEN_DIM = 256
EPOCHS = 10
NEG_RATIO = 4
BATCH_SIZE = 10

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X+list(data_description))
X_seq = tokenizer.texts_to_sequences(X)
des_seq = tokenizer.texts_to_sequences(list(data_description))

word_index = tokenizer.word_index
print ("Found %s unique tokens."%len(word_index))

data = pad_sequences(X_seq, maxlen=maxlen)
des = pad_sequences(des_seq, maxlen=maxlen)
lables = np.asarray(Y)

N = data.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices] 


## NEED a different way during testing
# x_train = data[: int(N*train_samples)]
# y_train = labels[: int(N*train_samples)]
# x_val = data[int(N*train_samples) : int(N*(train_samples+val_samples))]
# y_val = labels[int(N*train_samples) : int(N*(train_samples+val_samples))]
# x_test = data[int(N*(train_samples+val_samples)) : ]
# y_test = labels[int(N*(train_samples+val_samples)) : ]


#not using glove embeddings
def build_model():
	embedding_layer = Embedding(vocab_size,EMBED_DIM, input_length=maxlen)
	article_input = Input(shape=(maxlen,), dtype='int32')
	article_emb = embedding_layer(article_input)

	dataset_input = Input(shape=(maxlen,), dtype='int32')
	dataset_emb = embedding_layer(dataset_input)

	article_lstm = LSTM(HIDDEN_DIM)
	article_vector = article_lstm(article_emb)

	##vector: (batch_size, hidden_dim)

	dataset_lstm = LSTM(HIDDEN_DIM)
	dataset_vector = dataset_lstm(dataset_emb)

	merged = layers.merge.dot([article_vector, dataset_vector], axes=1)
	#(batch_size, 1)
	output = Dense(1, activation='sigmoid')(merged)

	model = Model([article_input, dataset_input], output)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model 

#each batch contains one postive 
#and several negative, ratio can adjust
def generate_batch(x_samples, y_samples, datasets, batch_size, neg_ratio=4):
	total_size = batch_size*(1+neg_ratio)
	num_batches = len(x_samples) // total_size
	while True:
		for bacthIdx in range(0, num_batches):
			start = batchIdx * total_size
			end = (batchIdx + 1)*total_size
			article_batch = np.zeros(shape=(total_size, maxlen))
			dataset_batch = np.zeros(shape=(total_size, maxlen))
			outputs = np.zeros(shape=(total_size,))
		
			for line in range(start, end):
				lineIdx = line - start 
				if (lineIdx + 1)%batch_size == 0:
					##add one positive 
					article_batch[lineIdx] = x_samples[line]
					##rmb id - 1 = idx
					dataset_idx = y_samples[line] - 1 
					dataset_batch[lineIdx] = datasets[dataset_idx]
					outputs[lineIdx] = 1

				else:
					##add one negative samples
					dataset_idx = np.random.randint(0, len(datasets))
					while dataset_idx == y_samples[line] - 1:
						dataset_idx = np.random.randint(0, len(datasets))
					article_batch[lineIdx] = x_samples[line]
					dataset_batch[lineIdx] = datasets[dataset_idx]
					outputs[lineIdx] = 0

			yield [article_batch, dataset_batch], outputs

def load_weights(self, weight_file_path):
    if os.path.exists(weight_file_path):
        model.load_weights(weight_file_path)

def get_weight_path(model_dir_path):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    return model_dir_path + '/' + MODEL_NAME + '-weights.h5'

def fit(model. X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, neg_ratio=NEG_RATIO, model_dir_path=None):
	if model_dir_path is None:
		model_dir_path = './models'
	weight_file_path = get_weight_path(model_dir_path)
	checkpoint = ModelCheckpoint(weight_file_path, save_best_only=True)
	train_gen = generate_batch(Xtrain, Ytrain, des, batch_size, NEG_RATIO)
	train_num_batches = len(Xtrain) // (batch_size*(1+NEG_RATIO))
	history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                  epochs=epochs, verbose=1, callbacks=[checkpoint])
    model.save_weights(weight_file_path)
    return history


model = build_model()
fit(model, data, labels)

##ISSUE: how to handle no mention case!!!

#predict one at a time
def predict(article, datasets):
	scores = []
	for dataset in datasets:
		scores.appens(model.predict(article, dataset))
	##return the index of dataset with the highest score
	return np.argmax(scores) + 1



