from nltk.tokenize import  WordPunctTokenizer, SpaceTokenizer
import numpy as np
import codecs


def read_sent(sent):
    sent = SpaceTokenizer().tokenize(sent)
    start = int(sent[0])
    end = int(sent[1])
    dataset = int(sent[2])
    sent = sent[4:]
    labels = [0]*len(sent)
    if dataset != 0:
        labels[start : end+1] = [1]*(end+1-start)
    return [(sent[i], str(labels[i])) for i in range(len(sent))]

def get_sents_by_dir(data_dir):
    data = open(data_dir, 'r')
    data_list = data.readlines()

    sentences = [read_sent(sent) for sent in data_list]

    data.close()
    return sentences

def get_sents(data):
    sentences = [read_sent(sent) for sent in data]
    return sentences

def read_doc(doc, labels):
    doc = SpaceTokenizer().tokenize(doc.strip())
    # doc = doc.strip().split()
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



## sample train and val data
def data_sampler(neg_ratio, val_ratio=0.025, data_dir='../../data/data_40/'):
    np.random.seed(2019)
    data = []
    with codecs.open(data_dir+'pos_data', 'r') as pos:
        data = pos.readlines()
    print (str(len(data))+' pos data sampled')
    with codecs.open(data_dir+'neg_data', 'r') as neg:
        neg_data = np.asarray(neg.readlines())
        neg_len = max(len(neg_data)-1, 0)
        neg_idx = np.random.random_integers(0, neg_len, int(neg_len*neg_ratio))
        data += list(neg_data[neg_idx])

    print (str(int(neg_len*neg_ratio))+' neg data sampled')
    np.random.shuffle(data)

    val = int(len(data)*(1-val_ratio))
    train_data = data[: val]
    val_data = data[val :]

    return train_data, val_data









