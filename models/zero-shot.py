import keras 
from keras import layers
from keras.layers import GlobalMaxPooling1D, Concatenate, Convolution1D, Embedding, Dropout, Dense, LSTM, Bidirectional, Input, Dense, Flatten, Permute, multiply, RepeatVector
from keras.layers import dot, Reshape, Multiply
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
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


def prep_data(neg_ratio=0.0125, neg_rate=3, val_ratio=0.05, data_dir='../../data/data_40/', maxlen=40, emb_dim=300):
    
    train_list, val_list = data_sampler(neg_ratio, val_ratio=0, data_dir=data_dir)

    train_sents, train_labels = data_loader(train_list)
#     val_sents, val_labels = data_loader(val_list)
    
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
#     X_val = tokenizer.texts_to_sequences(val_sents)

    X_train = pad_sequences(X_train, maxlen=maxlen)
#     X_val = pad_sequences(X_val, maxlen=maxlen)
    
    X_train, X_train_dataset, Y_train = convert_train_dataset(X_train, train_labels, neg_rate=neg_rate)
    total = len(X_train)
    val = int(total*val_ratio)
    X_val = X_train[ : val]
    X_val_dataset = X_train_dataset[ : val]
    Y_val = Y_train[ : val]
    
    X_train = X_train[val : ]
    X_train_dataset = X_train_dataset[val : ]
    Y_train = Y_train[val : ]
    
#     Y_train = np.asarray(train_labels)
#     Y_val = np.asarray(val_labels)
    
#     Y_train = keras.utils.to_categorical(Y_train, num_classes=DATASET_CLASS)
#     Y_val = keras.utils.to_categorical(Y_val, num_classes=DATASET_CLASS)

    return X_train, X_train_dataset, Y_train, X_val, X_val_dataset, Y_val, embedding_matrix, vocab_size, tokenizer


# convert dataset to new format
def convert_train_dataset(data_ids, labels, neg_rate=3):
        '''
        input:
        data: (N, maxlen) already tokenized to ids
        labels: N ids
        neg rate: each correct pair add 5 negative pairs
        output:
        data: N pairs (maxlen+datasetlen)
        pair_labels: 0, 1 (for training only)
        '''
        
        new_data = []
        new_dataset = []
        pair_labels = []
        for i in range(len(data_ids)):
            lb = labels[i]
            new_data.append(data_ids[i])
            new_dataset.append(dataset_name_idx[lb])
            pair_labels.append(1)
            
            chosen = [lb]
            for j in range(neg_rate):
                fake = np.random.choice(list(data_set_ids))
                while fake in chosen:
                    fake = np.random.choice(list(data_set_ids))
                chosen.append(fake)
                new_data.append(data_ids[i])
                new_dataset.append(dataset_name_idx[fake])
                pair_labels.append(0)
                
        new_data = np.asarray(new_data)
        new_dataset = np.asarray(new_dataset)
        pair_labels = np.asarray(pair_labels)
        
        # shuffle
        idx = [i for i in range(len(new_data))]
        np.random.shuffle(idx)
        new_data = new_data[idx]
        new_dataset = new_dataset[idx]
        pair_labels = pair_labels[idx]
        return new_data, new_dataset, pair_labels

def convert_test_dataset(data_ids, label):
    '''
    change data_ids to pairs, change labels to new label index
    process only one label 
    '''
    new_data = []
    new_dataset = []
    for i in range(len(data_ids)):
        new_data.append(data_ids[i])
        new_dataset.append(dataset_name_idx[label])
#     labels = [dataset2label[lb] for lb in labels]
    new_data = np.asarray(new_data)
    new_dataset = np.asarray(new_dataset)
    return new_data, new_dataset
    
    
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


def run(X_train, X_train_dataset, Y_train, X_val, X_val_dataset, Y_val, embedding_matrix, vocab_size, maxlen=40, max_dataset_len=20, emb_dim=300, kernel=64, window=5, drop=0.2, epochs=10):    
    mention_embedding = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    mention_input = Input(shape=(maxlen,))
    mention_emb = mention_embedding(mention_input)
    mention_emb = Dropout(drop)(mention_emb)
    
    dataset_embedding = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=max_dataset_len, trainable=False)
    dataset_input = Input(shape=(max_dataset_len,))
    dataset_emb = dataset_embedding(dataset_input)
    dataset_emb = Dropout(drop)(dataset_emb)
    
    mention_conv = Convolution1D(filters=kernel, kernel_size=window, activation='relu', kernel_initializer='random_uniform')
    dataset_conv = Convolution1D(filters=kernel, kernel_size=window, activation='relu', kernel_initializer='random_uniform')
    
    mention_vector = mention_conv(mention_emb)
    mention_vector = GlobalMaxPooling1D()(mention_vector) #(N, filters)
    
    dataset_vector = dataset_conv(dataset_emb)
    dataset_vector = GlobalMaxPooling1D()(dataset_vector) #(N, filters)
    
    merged = layers.merge.dot([mention_vector, dataset_vector], axes=1)
    output = Dense(1, activation = 'sigmoid')(merged)
    
    model = Model([mention_input, dataset_input], output)
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    earlyStop = [EarlyStopping(monitor='val_acc', patience=3)]
    history = model.fit([X_train, X_train_dataset], Y_train, batch_size=64, epochs=epochs, validation_split=0.01) 
    
    preds = model.predict([X_val, X_val_dataset]) 
    preds = [1 if y>=0.5 else 0 for y in preds]
    print (score(Y_val, preds, average=None, labels=[0, 1]))
#     p, r, f = classify_score(preds, Y_val)
    
    return model, history

def doc_pred(model, doc, tokenizer, MAXLEN=40):
    splits = []
    doc = doc.split()
    for i in range(0, len(doc), MAXLEN):
        splits.append(doc[i : i+MAXLEN])
    splits = tokenizer.texts_to_sequences(splits)
    splits = pad_sequences(splits, maxlen=MAXLEN)
    all_preds = [] # (DATASET_CLASS, N)
    for label in [0] + list(data_set_ids):
        splits_data, splits_dataset = convert_test_dataset(splits, label)
        preds = model.predict([splits_data, splits_dataset])
        preds = np.squeeze(preds)
        all_preds.append(preds)
    all_preds = np.asarray(all_preds)
    preds = np.argmax(all_preds, axis=0) # (N)
    preds = [label2dataset[y] for y in preds]
    return preds

def doc_eval(model, doc, labels, tokenizer, MAXLEN=40):
    preds = []
    counter = 0
    for d in tqdm(doc):
        counter += 1
        cur_preds = doc_pred(model, d, tokenizer, MAXLEN)
        if counter%10 == 0:
            print (cur_preds)
        preds.append(cur_preds)
    p, r, f = classify_score(preds, labels)
    return p, r, f


if __name__=='__main__':
	##load glove
	embedding_index = {}
	f = open('../../glove.840B.300d.txt')
	for line in f:
	    values = line.split()
	    word = ''.join(values[:-300])
	    coefs = np.asarray(values[-300:], dtype='float32')
	    embedding_index[word] = coefs
	f.close()


	# dataset base
	DIR = "../../train_test/"
	data_set_citations = pd.read_json(DIR+'data_set_citations.json', encoding='utf-8')
	data_set_ids = data_set_citations['data_set_id'].values
	data_set_ids = set(data_set_ids)

	dataset2label = {}
	label2dataset = {}
	dataset2label[0] = 0 # 0 for no dataset
	label2dataset[0] = 0
	counter = 1 
	for d in data_set_ids:
	    dataset2label[d] = counter
	    label2dataset[counter] = d 
	    counter += 1
	DATASET_CLASS = len(dataset2label)


	# load dataset meta info
	datasets = pd.read_json('../../train_test/data_sets.json', encoding='utf-8')
	dataset_name = datasets['name'].values
	# dataset_subjects = datasets['subjects'].values
	# dataset_description = datasets['description'].values
	# dataset_mentions = datasets['mention_list'].values

	MAX_NAME_LEN = 20
	dataset_name= [name.lower() for name in dataset_name]
	dataset_name = ['there is no dataset'] + dataset_name
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(dataset_name)
	dataset_name_idx = tokenizer.texts_to_sequences(dataset_name)
	dataset_name_idx = pad_sequences(dataset_name_idx, maxlen=MAX_NAME_LEN)


	neg_ratio = 0.002
	# not sampling any dev data
	X_train, X_train_dataset, Y_train, X_val, X_val_dataset, Y_val, embedding_matrix, vocab_size, tokenizer = prep_data(neg_ratio=neg_ratio, neg_rate=4, val_ratio=0.05) 
	# will not do eval on dev to save time
	test_labels, test_sents, zeroshot_labels, zeroshot_sents = load_test()

	model, history = run(X_train, X_train_dataset, Y_train, X_val, X_val_dataset, Y_val, embedding_matrix, vocab_size, maxlen=40, max_dataset_len=20, emb_dim=300, kernel=256, window=6, drop=0.2, epochs=5) 
	doc_eval(model, test_sents, test_labels, tokenizer)
	doc_eval(model, zeroshot_sents, zeroshot_labels, tokenizer)
















