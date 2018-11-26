import keras 
from keras import layers
from keras.layers import Embedding, Dense, LSTM, Bidirectional, Input, Dense, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import ModelCheckpoint
import numpy as np 
import json
import pandas as pd 
import os

data_set = pd.read_json('../../train_test/data_sets.json', encoding='utf-8')
#note: dataset_id = index + 1
data_description = data_set["description"].values

DIR = '../../data/golden_data'

X = []
Y = []

with open(DIR, 'r') as f:
	for line in f:
		line = line.strip().split()
		Y.append(int(line[0]))
		X.append(' '.join(line[1:]))

print (len(X), 'sampled loaded')

##X: strings of texts
##Y: dataset id mentioned in that string

# Add a sentence for no mention case
data_description = list(data_description)
data_description.insert(0, "There is no mention.")

maxlen = 200
vocab_size = 50000 ##more than 80K unique tokens
EMB_DIM = 50
HIDDEN_DIM = 256
EPOCHS = 10
NEG_RATIO = 4
BATCH_SIZE = 10
DATASET_CLASS = len(data_description) 
MODEL_NAME = "LSTM"

#actual batch size = BATCH_SIZE * (1 + NEG_RATIO)


tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X+data_description)
X_seq = tokenizer.texts_to_sequences(X)
des_seq = tokenizer.texts_to_sequences(list(data_description))

word_index = tokenizer.word_index

data = pad_sequences(X_seq, maxlen=maxlen)
des = pad_sequences(des_seq, maxlen=maxlen)
labels = np.asarray(Y)

##randomly shuffle data and labels
##np.random.seed(0)
N = data.shape[0]
indices = np.arange(N)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices] 


## I set apart the last 2000 samples as test set
test_data = data[-2000 : ]
test_labels = labels[-2000 : ]
data = data[ : -2000]
labels = labels[ : -2000]

## I am not using Glove here, may get better results with Glove
def build_model():
    embedding_layer = Embedding(vocab_size, EMB_DIM, input_length=maxlen)
    article_input = Input(shape=(maxlen,), dtype='int32')
    article_emb = embedding_layer(article_input)
    
    dataset_input = Input(shape=(maxlen,), dtype='int32')
    dataset_emb = embedding_layer(dataset_input)
    
    article_lstm = LSTM(HIDDEN_DIM)
    article_vector = article_lstm(article_emb)
    #vector shape: (batch_size, hidden_dim)
    
    dataset_lstm = LSTM(HIDDEN_DIM)
    dataset_vector = dataset_lstm(dataset_emb)
    
    merged = layers.merge.dot([article_vector, dataset_vector], axes=1)
    #shape: (batch_size, 1)
    output = Dense(1, activation='sigmoid')(merged)
    
    model = Model([article_input, dataset_input], output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

##the batch for each x_sample contains one correct match,
##NEG_RATIO wrong matches, (negative sampling)
##the batch_size here means how many different x_samples in one batch
def generate_batch(x_samples, y_samples, datasets, batch_size, neg_ratio=4):
    total_size = batch_size*(1+NEG_RATIO)
    num_batches = len(x_samples) // batch_size
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            article_batch = np.zeros(shape=(total_size, maxlen))
            dataset_batch = np.zeros(shape=(total_size, maxlen))
            outputs = np.zeros(shape=(total_size,))
            lineIdx = 0 ##index in the batch 
            
            ## fill in one batch
            for line in range(start, end):
                #each x is used (1+neg_ratio) times
                for i in range(1+NEG_RATIO):
                    if i == 0:
                        ## Add one correct match
                        article_batch[lineIdx] = x_samples[line]
                        dataset_idx = y_samples[line]
                        dataset_batch[lineIdx] = datasets[dataset_idx]
                        outputs[lineIdx] = 1
                        lineIdx += 1
                    else:
                        dataset_idx = np.random.randint(0, DATASET_CLASS)
                        while dataset_idx == y_samples[line]:
                            dataset_idx = np.random.randint(0, DATASET_CLASS)
                        article_batch[lineIdx] = x_samples[line]
                        dataset_batch[lineIdx] = y_samples[dataset_idx]
                        outputs[lineIdx] = 0
                        lineIdx += 1
            
            ##can shuffle the batch here as well
            yield [article_batch, dataset_batch], outputs
    
    
def load_weights(model, weight_file_path):
    if os.path.exists(weight_file_path):
        model.load_weights(weight_file_path)

def get_weight_path(model_dir_path):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    return model_dir_path + '/' + MODEL_NAME + '-weights.h5'


def fit(model, X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, neg_ratio=NEG_RATIO, model_dir_path=None):  
    if model_dir_path is None:
        model_dir_path = '../../models'
    weight_file_path = get_weight_path(model_dir_path)
    checkpoint = ModelCheckpoint(weight_file_path, save_best_only=True)
    train_gen = generate_batch(data, labels, des, batch_size, neg_ratio)
    train_num_batches = len(data) // batch_size
    history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                  epochs=epochs, verbose=1, callbacks=[checkpoint])
    model.save_weights(weight_file_path)
    return history
    

#retuen the predicted mention and the confidence
#label 0 means no mention
def inference(test_data, datasets):
    scores = []
    labels = []
    for x in test_data:
        max_score = 0
        max_index = 0
        for i in range(len(datasets)):
            ##batch_size here is 1
            s = model.predict([[x], [data[i]]])
            if s > max_score:
                max_score = s
                max_index = i
        scores.append(max_score)
        labels.append(max_index)
    return scores, labels


def evaluate(outputs, targets):
    precision, recall, fscore, support = score(targets, output)
    
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))

    return precision, recall, fscore


if __name__ == '__main__':
	model = build_model()
	fit(model, data, labels)

	model.load_weights(get_weight_path('../../models'))
	scores, labels = inference(test_data, des)

	p, r, f = evaluate(labels, test_labels)
















