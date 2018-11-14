import pandas as pd 
import os, sys, random 
from nltk.tokenize import WordPunctTokenizer
from collections import Counter 
import json, codecs 

#this is where I saved my data,
#change it if you stored in another dir
DIR = "../../train_test/"

class DataParser:
	def __init__(self, outdir=None):
		self.data_set_citations = pd.read_json(DIR+'data_set_citations.json', encoding='utf-8')
		self.data_set = pd.read_json(DIR+'data_sets.json', encoding='utf-8')
		self.publications = pd.read_json(DIR+'publications.json', encoding='utf-8')

		self.publications.insert(6, 'full_text', pd.Series)
		self.full_text = self._extract()
		full_text_series = pd.Series()

		if not os.path.isdir(outdir):
			os.makedirs(outdir)
		self.outdir = outdir


		count = 0 
		for i, file_id in zip(self.publications.index, self.publications['publication_id']):
			try:
				full_text_series.loc[i] = self.full_text[str(file_id)+'.txt']
				count += 1
			except:
				if (type(file_id)!=str):
					print (type(file_id))
				print(str(file_id) + '.txt not found in files')
				pass

		print ("{} files added".format(count))
		self.publications['full_text'] = full_text_series 

	
	def get_train_data_full(self):
		print ('Generating gold data file')
		#max length of each sample
		#I use 200 here, Animesh used 500,
		#shorter should be easier for LSTM 
		max_length_token = 200
		#Animesh used ratio 2, I am using 5 here 
		#to have more no mention cases
		pos_neg_sample_ratio = 5
		pos_count = 0
		neg_count = 0
		zero_count = 0

		with codecs.open(self.outdir+'golden_data', 'w+') as golden_data:
			for index, row in self.data_set_citations.iterrows():
				#get the relevant publciaiton info
				sample_row = self.publications.loc[self.publications['publication_id']==row['publication_id']]
				sample_text = sample_row['full_text'].values[0]
				sample_text_tokens = list(WordPunctTokenizer().tokenize(sample_text))
				sample_text_spans = list(WordPunctTokenizer().span_tokenize(sample_text))
				dataset_id = row['data_set_id']

				pos_splits = [] 
				for mention in row['mention_list']:
					mention_text = mention 
					mention_text_spans = list(WordPunctTokenizer().span_tokenize(mention_text))

					find_index = sample_text.find(mention_text)

					try:
						## found this mention in the paper
						if find_index != -1:
							mention_text_spans = [(indices[0]+find_index, indices[1]+find_index) for indices in mention_text_spans]

							#find a split containing this mention
							#only store first match
							for splits in range(len(sample_text_tokens)//(max_length_token) -1):
								if sample_text_spans.index(mention_text_spans[0]) >= splits*max_length_token and \
									sample_text_spans.index(mention_text_spans[-1]) < (splits+1)*max_length_token:

									pos_splits.append(splits)

									#I did not include the pointer positions here
									#since my baseline does not really need them
									golden_data.write(
										str(dataset_id) + ' '+
										' '.join(sample_text_tokens[splits*(max_length_token):(splits+1)*max_length_token])
										+'\n')
									pos_count += 1
									break 
						else:
							neg_count += 1

					except:
						pass 
						
				for splits in range(len(sample_text_tokens)//max_length_token -1):
					if splits not in pos_splits and random.randint(0, 100) < pos_neg_sample_ratio:
						golden_data.write(
							str(0) + ' ' +
							' '.join(sample_text_tokens[splits*(max_length_token):(splits+1)*max_length_token])
							+'\n')
						zero_count += 1
		print (pos_count, "positive samples added")
		print (neg_count, "samples not found")
		print (zero_count, 'no mention samples added')
		print ('\n')

	def get_vocab(self, start_index=2, min_count=10):
		print ('Getting vocab')
		#print (self.publications['full_text'].values[0])
		text = ' '.join(list(self.publications['full_text'].values))
		all_words = WordPunctTokenizer().tokenize(text+text.lower())
		vocab = Counter(all_words).most_common()
		vocab_out_json = {}
		for items in vocab:
			if items[1] > min_count:
				assert type(items[0]) == str 
				vocab_out_json[items[0]] = len(vocab_out_json) + start_index

		print (len(vocab) - len(vocab_out_json), ' words are discarded as OOV')
		print (len(vocab_out_json), ' words are in vocab')

		with codecs.open(self.outdir+'vocab.json', 'w+') as vocabfile:
			json.dump(vocab_out_json, vocabfile)



	def _extract(self, dir_name=DIR+'files/text/', extension='.txt'):
		full_text = {}
		for item in os.listdir(dir_name):
			if item.endswith(extension):
				file_name = os.path.abspath(dir_name+'/'+item)
				with codecs.open(file_name, 'r') as f:
					try:
						lines = f.readlines()
						text = ' '.join([s.strip() for s in lines])
						text.replace('-', '')
						full_text[item] = text
					except:
						print ('When extracting, Did not find ', str(item))
		return full_text 

if __name__ == '__main__':
	data_parser = DataParser(outdir='../../data/')
	#data_parser.get_vocab()
	data_parser.get_train_data_full()














