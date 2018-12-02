import codecs
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

class OuputEvaluator:

    def __init__(self, indir=None, outdir=None):
        self.indir = indir
        self.outdir = outdir


    # TODO: doc level evaluation here

    def generate_report(self, ground_truth='test.txt', model_out='model_test.txt'):
        with codecs.open(self.indir + ground_truth, 'rb') as ground_split, \
            codecs.open(self.outdir + model_out, 'rb') as model_out_split:
            true_lines = ground_split.readlines()
            predict_lines = model_out_split.readlines()

            def compute_partial(s1,s2,e1,e2):
                tp = min(max(s1,s2) - min(e1, e2), 0)
                total_predicted = (e1-s1) + (e2-s2)
                try:
                    partial_accuracy = float(tp)/total_predicted
                except:
                    partial_accuracy = 0
                try:
                    partial_precision = float(tp)/ (e2-s2)
                except:
                    partial_precision = 0
                try:
                    partial_recall = float(tp)/ (e1-s1)
                except:
                    partial_recall = 0
                try:
                    partial_fscore = 2 * (partial_precision * partial_recall) / (partial_precision + partial_recall)
                except:
                    partial_fscore = 0
                return partial_accuracy, partial_precision, partial_recall, partial_fscore

            def match_report():
                true_mention = []
                predict_mention = []
                true_dataset = []
                predict_dataset = []

                partial_accuracy = []
                partial_precision = []
                partial_recall = []
                partial_fscore = []


                for trues, predicts in zip(true_lines, predict_lines):
                    multi_predicts = predicts.split('|')
                    single_true = trues.split()
                    for i in range(len(multi_predicts)):
                        predicts = [int(t) for t in multi_predicts[i].split()[:4]]
                        trues = [int(t) for t in single_true[:4]]
                        print (predicts, trues)
                    #print (trues, predicts)

                    if predicts[0] != -1:
                        true_mention.append(1)
                        if trues[0] == predicts[0] and trues[1] == predicts[1]:
                            predict_mention.append(1)
                        else:
                            predict_mention.append(0)

                        a,b,c,d = compute_partial(trues[0],predicts[0],trues[1],predicts[1])
                        partial_accuracy.append(a)
                        partial_precision.append(b)
                        partial_recall.append(c)
                        partial_fscore.append(d)

                    if predicts[2] != -1:
                        true_dataset.append(trues[2])
                        predict_dataset.append(predicts[2])


                P, R, F, S = precision_recall_fscore_support(true_mention, predict_mention)
                print(P, R, F, S)
                #print (P[1], R[1], F[1])

                P, R, F, S = precision_recall_fscore_support(true_dataset, predict_dataset)
                print(np.average(P), np.average(R), np.average(F), np.average(S))

                A, P, R, F = np.average(partial_accuracy), np.average(partial_precision), \
                             np.average(partial_recall), np.average(partial_fscore)
                print (P, R, F, A)

            match_report()



if __name__ == "__main__":
    o = OuputEvaluator('data/', 'output/')
    o.generate_report(ground_truth='test.txt',
                      model_out='BiLSTM_CRF_preds'
    )