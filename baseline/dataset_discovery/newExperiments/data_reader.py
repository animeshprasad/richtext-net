def read_sent(sent):
	sent = sent.strip().split()
	start = int(sent[0])
	end = int(sent[1])
	dataset = int(sent[2])
	sent = sent[4:]
	labels = [0]*len(sent)
	if dataset != 0:
		labels[start : end+1] = [1]*(end+1-start)
	return [(sent[i], str(labels[i])) for i in range(len(sent))]

def get_sents(data_dir):
	data = open(data_dir, 'r')
	data_list = data.readlines()

	sentences = [read_sent(sent) for sent in data_list]

	data.close()
	return sentences

def read_doc(doc, labels):
    doc = doc.strip().split()
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

