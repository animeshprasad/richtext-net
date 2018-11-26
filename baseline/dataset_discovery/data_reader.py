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

	data.close()
	return sentences

		
