import pandas as pd
import os, sys, random
from nltk.tokenize import  TreebankWordTokenizer, SpaceTokenizer
from collections import Counter
import json, codecs, re
import logging
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

## default parameters:
# MAX_LENGTH=30
# DIR = "../../train_test/"

def findall_lower(p, s):
    i = s.lower().find(p.lower())
    while i != -1:
        yield i
        i = s.lower().find(p.lower(), i + 1)


class DataParser:
    def __init__(self, outdir=None, DIR = "../train_test/"):
        ## outdir: where to store the sampled data
        ## DIR: directory where you store the data
        self.DIR = DIR
        self.data_set_citations = pd.read_json(DIR+'data_set_citations.json', encoding='utf-8')
        self.full_text = self._extract()
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir) 


    def get_train_data(self,  MAX_LENGTH=60, full_neg=True):
        ## MAX_LENGTH: max length of segments to be split into
        ## neg ratio: how many neg data to use (out of 100), should be an integer
        ## full_neg: whether to extract all neg data

        max_length_token = MAX_LENGTH

        ## avoid taking docs from test set
        test_doc_ids = []
        zero_shot_doc_ids = []
        with open('../data/all_test_docs/test_doc_ids') as f:
            fl = f.readlines()
            test_doc_ids = [int(line.strip()) for line in fl]

        with open('../data/all_test_docs/zero_shot_doc_ids') as f:
            fl = f.readlines()
            zero_shot_doc_ids = [int(line.strip()) for line in fl]

        train_doc_len = len(set(self.data_set_citations['publication_id'].values)) - len(test_doc_ids) - len(zero_shot_doc_ids)
        logger.info('sample from '+str(train_doc_len)+' train docs')

        pos_count = 0
        neg_count = 0
        pos_tokens = 0
        neg_tokens = 0

        sampled = []
        with codecs.open(self.outdir + 'pos_data', 'w') as pos_data, codecs.open(self.outdir+'neg_data', 'w') as neg_data:
            for index, row in self.data_set_citations.iterrows():
                pub_id = row['publication_id']
                if pub_id in zero_shot_doc_ids or pub_id in test_doc_ids:
                    continue

                if pub_id in sampled:
                    continue
                else:
                    sampled.append(pub_id)

                pub_ids = list(self.data_set_citations['publication_id'])
                rows = [pub_ids.index(i) for i in pub_ids if i==pub_id]
                mention_list = []
                for r in rows:
                    d_row = self.data_set_citations.loc[r]
                    mention_list.extend(d_row['mention_list'])
                mention_list = set(mention_list)
                logger.info('pub id: {}, mentions: {}'.format(pub_id, len(mention_list)))
    
                sample_text = self.full_text[str(pub_id)+'.txt']
                sample_text_tokens = list(SpaceTokenizer().tokenize(sample_text))
                sample_text_spans = list(SpaceTokenizer().span_tokenize(sample_text))

                pos_splits = []
                for mention_text in mention_list:
                    mention_text = re.sub('\d', ' ', mention_text)
                    # mention_text = re.sub('[^ ]- ', '', mention_text)
                    mention_text_spans = list(SpaceTokenizer().span_tokenize(mention_text))

                    index_finder_lower = findall_lower(mention_text, sample_text)

                    all_found_indices = [idx for idx in index_finder_lower]

                    for find_index in all_found_indices:
                      try:
                        if find_index != -1:
                            # logger.info('Found: '+mention_text)
                            new_mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]
                            #write to training sample pointers here

                            for splits in range(len(sample_text_tokens) // max_length_token - 1):
                                if sample_text_spans.index(new_mention_text_spans[0]) > splits*(max_length_token) and \
                                  sample_text_spans.index(new_mention_text_spans[-1]) < (splits+1)*(max_length_token):

                                    pos_splits.append(splits)
                                    pos_count += 1
                                    pos_tokens += len(new_mention_text_spans)
                                    neg_tokens += (MAX_LENGTH - len(new_mention_text_spans))

                                    #TODO Wrapper over full data reader
                                    pos_data.write(
                                        str(sample_text_spans.index(new_mention_text_spans[0]) - splits*(max_length_token)) +
                                        ' ' + str(sample_text_spans.index(new_mention_text_spans[-1]) - splits*(max_length_token)) +
                                        ' ' + str(row['data_set_id']) + ' ' + str(row['publication_id']) +
                                         ' ' + ' '.join(sample_text_tokens[splits*(max_length_token):(splits+1)*(max_length_token)])
                                        + '\n'
                                    )
                        else:
                            # print ('Annotation Error: Annotated gold standards not correct')
                            pass
                      except:
                        # print ('Indexing Logic Error: Some corner index case missed while parsing')
                        pass


                if not full_neg:
                    continue
                ## NOTE: index starts from 0
                ## -1 - 1 means no mention
                for splits in range(len(sample_text_tokens) // (max_length_token) - 1):
                    if splits not in pos_splits:
                        neg_data.write(
                            str(-1) + ' ' + str(-1) +
                            ' ' + str(0) + ' ' + str(row['publication_id']) +
                            ' ' + ' '.join(sample_text_tokens[splits * (max_length_token):(splits + 1) * (
                                                max_length_token)])
                            + '\n'
                        )

                        neg_count += 1
                        neg_tokens += MAX_LENGTH

        logger.info(str(pos_count)+" mentions added.")
        logger.info(str(neg_count)+" no mentions added.")
        logger.info(str(pos_tokens)+" pos tokens added.")
        logger.info(str(neg_tokens)+" neg tokens added.")
        logger.info("neg token percentage: {}".format(neg_tokens*100/(pos_tokens+neg_tokens)))

        # train = 0
        # val = 0
        # with codecs.open(self.outdir + 'golden_data', 'r') as golden_data, \
        #     codecs.open(self.outdir + 'train.txt', 'w') as train_split, \
        #     codecs.open(self.outdir + 'validate.txt', 'w') as validate_split:
        #     all_lines = golden_data.readlines()
        #     for i, line in enumerate(all_lines):
        #         if i%10 == 0:
        #             validate_split.write(line)
        #             val += 1
        #         else:
        #             train_split.write(line)
        #             train += 1

        # logger.info(str(train)+' training segments sampled')
        # logger.info(str(val)+' validation segments sampled')



    def get_vocab(self, start_index=2, min_count=10):
        text = ''.join(list(self.publications['full_text'].values))
        all_words = SpaceTokenizer().tokenize(text + text.lower())
        vocab = Counter(all_words).most_common()
        vocab_out_json = {}
        for items in vocab:
            if items[1] > min_count:
                vocab_out_json[items[0].decode('utf-8', 'replace')] = len(vocab_out_json) + start_index

        print(len(vocab) - len(vocab_out_json), ' words are discarded as OOV')
        print (len(vocab_out_json), ' words are in vocab')

        with codecs.open(self.outdir + 'vocab.json', 'wb') as vocabfile:
            json.dump(vocab_out_json, vocabfile)



    def _extract(self, dir_name='files/text/', extension='.txt'):
        dir_name = self.DIR + dir_name
        full_text = {}
        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with codecs.open(file_name, 'r') as f:
                    try:
                        lines = f.readlines()
                        #TODO document structure
                        #text = ' '.join([s.strip() for s in lines])
                        text = ' '.join([s.strip() for s in lines])
                        text = re.sub('\d', ' ', text)
                        text = re.sub('[^ ]- ', '', text)
                        full_text[item] = text
                    except:
                        pass
        return full_text


class TestDataGenerator:
    ## generate and store zero_shot and non_zero_shot test docs
    def __init__(self, outdir='../data/test_docs', DIR='../train_test/'):
        self.DIR = DIR
        self.outdir = outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.data_set_citations = pd.read_json(DIR+'data_set_citations.json', encoding='utf-8')
        # self.publications = pd.read_json(DIR+'publications.json', encoding='utf-8')
        self.full_text = self._extract() 
        self.zero_shot_doc_ids = []
                
        self.publication_dataset = defaultdict(list)
        for i in range(len(self.data_set_citations)):
            row = self.data_set_citations.loc[i]
            self.publication_dataset[row['publication_id']].append(row['data_set_id'])   
        logger.info(str(len(self.publication_dataset)) + ' publications loaded')     

        ## for reproductivity
        ## same seed ensures same test set generated
        np.random.seed(2019)
        
        
    def _extract(self, dir_name='files/text/', extension='.txt'):
        dir_name = self.DIR + dir_name
        full_text = {}
        for item in os.listdir(dir_name):
            if item.endswith(extension):
                file_name = os.path.abspath(dir_name + '/' + item)
                with codecs.open(file_name, 'r') as f:
                    try:
                        lines = f.readlines()
                        text = ' '.join([s.strip() for s in lines])
                        text = re.sub('\d', ' ', text)
                        text = re.sub('[^ ]- ', '', text)
                        full_text[item] = text
                    except:
                        pass
        return full_text
    
    def get_zero_shot_docs(self):
        zero_shot_doc_ids = open(self.outdir+'/zero_shot_doc_ids', 'w+')
        zero_shot_docs = open(self.outdir+'/zero_shot_docs', 'w+')
        golden_data = open(self.outdir+'/zero_shot_doc_gold', 'w+')
        zero_shot_dataset_ids = open(self.outdir+'/zero_shot_dataset_ids', 'w+')
        
        zero_shot_dataset = []
        for dataset in set(self.data_set_citations['data_set_id']):
            if np.random.randint(0, 100) < 7:
                zero_shot_dataset.append(dataset)
                zero_shot_dataset_ids.write(str(dataset)+'\n')
        logger.info(str(len(zero_shot_dataset))+' zero-shot datasets selected')
        
        zero_shot_pub_ids = []
        for k, v in self.publication_dataset.items():
            ## all docs containing these datasets are set apart
            for data in zero_shot_dataset:
                if data in v:
                    zero_shot_pub_ids.append(k)
                    zero_shot_doc_ids.write(str(k)+'\n')
                    break
        logger.info(str(len(zero_shot_pub_ids)) + ' zero-shot docs selected')
        self.zero_shot_doc_ids = zero_shot_pub_ids
        
        pub_ids = list(self.data_set_citations['publication_id'])
        pos_tokens = 0
        neg_tokens = 0
        #to locate lines with relevant pubs
        for pub_id in zero_shot_pub_ids:
            pub_text = self.full_text[str(pub_id)+'.txt']
            zero_shot_docs.write(pub_text+'\n')
            pub_text_tokens = list(SpaceTokenizer().tokenize(pub_text))
            pub_text_spans = list(SpaceTokenizer().span_tokenize(pub_text))
            cur_pos_tokens = 0
            cur_neg_tokens = len(pub_text_tokens)
            
            res_line = []
            rows = [pub_ids.index(i) for i in pub_ids if i==pub_id]
            for idx in rows:
                d_row = self.data_set_citations.loc[idx]
                for mention_text in d_row['mention_list']:
                    mention_text = re.sub('\d', ' ', mention_text)
                    # mention_text = re.sub('[^ ]- ', '', mention_text)
                    mention_text_spans = list(SpaceTokenizer().span_tokenize(mention_text))
                    
                    index_finder_lower = findall_lower(mention_text, pub_text)
                    found_indices = [idx for idx in index_finder_lower]
                    
                    for find_index in found_indices:
                        try:
                            if find_index != -1:
                                cur_pos_tokens += len(mention_text_spans)
                                new_mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]

                                res_line.append((pub_text_spans.index(new_mention_text_spans[0]), 
                                                 pub_text_spans.index(new_mention_text_spans[-1]), 
                                                 d_row['data_set_id'], d_row['publication_id']))
                        except:
                            pass
            res_line = list(set(res_line))
            if len(res_line)==0:
                # no mentions at all
                res_line.append((-1, -1, 0, pub_id))
            i = 0
            for c in res_line:
                if i > 0:
                    golden_data.write(' | '+str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
                else:
                    golden_data.write(str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
                i+=1
            golden_data.write('\n')   
            pos_tokens += cur_pos_tokens
            neg_tokens += (cur_neg_tokens - cur_pos_tokens)

        zero_shot_doc_ids.close()
        zero_shot_docs.close()
        golden_data.close()
        zero_shot_dataset_ids.close()

        logger.info(str(pos_tokens)+" pos tokens added.")
        logger.info(str(neg_tokens)+" neg tokens added.")
        logger.info("neg token percentage: {}".format(neg_tokens*100/(pos_tokens+neg_tokens)))
    
    def get_test_docs(self):
        test_doc_ids = open(self.outdir+'/test_doc_ids', 'w+')
        test_docs = open(self.outdir+'/test_docs', 'w+')
        golden_data = open(self.outdir+'/test_doc_gold', 'w+')

        test_doc_list = []
        for doc in set(self.data_set_citations['publication_id']):
            if np.random.randint(0, 100) < 10 and doc not in self.zero_shot_doc_ids:
                test_doc_list.append(doc)
                test_doc_ids.write(str(doc)+'\n')
        logger.info(str(len(test_doc_list)) + ' test docs selected')
        
        
        pub_ids = list(self.data_set_citations['publication_id'])
        pos_tokens = 0
        neg_tokens = 0
        #to locate lines with relevant pubs
        for pub_id in test_doc_list:
            pub_text = self.full_text[str(pub_id)+'.txt']
            test_docs.write(pub_text+'\n')
            pub_text_tokens = list(SpaceTokenizer().tokenize(pub_text))
            pub_text_spans = list(SpaceTokenizer().span_tokenize(pub_text))
            cur_pos_tokens = 0
            cur_neg_tokens = len(pub_text_tokens)
            
            res_line = []
            rows = [pub_ids.index(i) for i in pub_ids if i==pub_id]
            for idx in rows:
                d_row = self.data_set_citations.loc[idx]
                for mention_text in d_row['mention_list']:
                    mention_text = re.sub('\d', ' ', mention_text)
                    # mention_text = re.sub('[^ ]- ', '', mention_text)
                    mention_text_spans = list(SpaceTokenizer().span_tokenize(mention_text))
                    
                    index_finder_lower = findall_lower(mention_text, pub_text)
                    found_indices = [idx for idx in index_finder_lower]
                    
                    for find_index in found_indices:
                        try:
                            if find_index != -1:
                                new_mention_text_spans = [(indices[0] + find_index, indices[1] + find_index) for indices in mention_text_spans]
                                cur_pos_tokens += len(mention_text_spans)

                                res_line.append((pub_text_spans.index(new_mention_text_spans[0]), 
                                                 pub_text_spans.index(new_mention_text_spans[-1]), 
                                                 d_row['data_set_id'], d_row['publication_id']))
                        except:
                            pass
            res_line = list(set(res_line))
            if len(res_line)==0:
                # no mentions at all
                res_line.append((-1, -1, 0, pub_id))
            i = 0
            for c in res_line:
                if i > 0:
                    golden_data.write(' | '+str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
                else:
                    golden_data.write(str(c[0])+' '+str(c[1])+' '+str(c[2])+' '+str(c[3]))
                i+=1
            golden_data.write('\n')   
            pos_tokens += cur_pos_tokens
            neg_tokens += (cur_neg_tokens - cur_pos_tokens)
        
        test_doc_ids.close()
        test_docs.close()
        golden_data.close()

        logger.info(str(pos_tokens)+" pos tokens added.")
        logger.info(str(neg_tokens)+" neg tokens added.")
        logger.info("neg token percentage: {}".format(neg_tokens*100/(pos_tokens+neg_tokens)))
        

if __name__ == '__main__':
    # test_parser = TestDataGenerator(outdir='../data/all_test_docs')
    # test_parser.get_zero_shot_docs()
    # test_parser.get_test_docs()

    # data_parser = DataParser(outdir='../data/data_20/')
    # data_parser.get_train_data(20, False)   

    # data_parser = DataParser(outdir='../data/data_40/')
    # data_parser.get_train_data(40, False)   

    data_parser = DataParser(outdir='../data/data_80/')
    data_parser.get_train_data(80, False)  

    data_parser = DataParser(outdir='../data/data_40/')
    data_parser.get_train_data(40)   





'''
Some subtlety in sampling the data:
ASSUME each train segment only contains one true mention, opposite cases should be quite rare.

there are 2 ways to evaluate, i.e. set apart some test docs and set apart some datasets and all docs containing these test datasets. 
Some subtlety here: if we remove all docs containing test datasets from the training set, 
these test sets may still contain other datasets that are not test datasets and appeared in training. 
On the other hand, those test docs we directly set apart may have some zero-shot cases where all mentions of that dataset are in the test set. 
However, such cases should be quite rare, so this should be fine and there should be a clear difference between the results on these two test sets.

IMPORTANT: use nltk SpaceTokenizer in training as well so that the index will all be consistent (same as split by space)
'''


















