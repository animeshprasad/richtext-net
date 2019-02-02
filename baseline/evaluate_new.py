# A match is counted as correct if 100% overlap with truth
def discovery_exact_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_preds = 0
            total_n = len(pred_list)
            correct = 0

            for i in range(total_n):
                true = true_list[i].strip().split()
                t_start, t_end = int(true[0]), int(true[1])

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for j in range(len(preds)):
                    total_preds += 1
                    if preds[j][0]==t_start and preds[j][1]==t_end:
                        correct += 1
                        break
            return correct / total_preds


# A match is counted as correct if there is at least 50% overlap 
# with the ground truth
# can change to other percentage as well
def discovery_partial_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_preds = 0
            total_n = len(pred_list)
            correct = 0

            for i in range(total_n):
                true = true_list[i].strip().split()
                t_start, t_end = int(true[0]), int(true[1])

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for j in range(len(preds)):
                    total_preds += 1
                    if (preds[j][0]<=t_start and preds[j][1]>=t_start+(t_end-t_start)//2) \
                        or (preds[j][0]<=t_start+(t_end-t_start)//2 and preds[j][1]>=t_end) \
                        or (preds[j][0]>=t_start and preds[j][1]<=t_end and (preds[j][1]-preds[j][0])>=(t_end-t_start)//2):
                        correct += 1
                        break
            return correct / total_preds


# doc level exact match
# correct preds / total preds
# def doc_exact_match(pred_file, truth_file):
#     with open(pred_file, 'r') as fp:
#         with open(truth_file, 'r') as ft:
#             pred_list = fp.readlines()
#             true_list = ft.readlines()
#             assert len(pred_list) == len(true_list), "Test cases mismatch"
#             total_n = len(pred_list)  ## total number of docs
#             total_pred = 0  ## total number of preds
#             correct = 0

#             for i in range(total_n):
#                 true = true_list[i].strip().split('|')
#                 true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

#                 preds = pred_list[i].strip().split('|')
#                 preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

#                 for p in range(len(preds)):
#                     total_pred += 1
#                     for t in true:
#                         if preds[p][0]==t[0] and preds[p][1]==t[1]:
#                             correct += 1
#                             break

#             return correct / total_pred

def doc_exact_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_n = len(pred_list)  ## number of docs
            total_pred = 0  ## number of preds
            total_true = 0  ## number of truth
            precision_correct = 0 ## pred in truth

            for i in range(total_n):
                true = true_list[i].strip().split('|')
                true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for p in range(len(preds)):
                    total_pred += 1
                    for t in true:
                        if preds[p][0]==t[0] and preds[p][1]==t[1] and t[0]!=-1 and t[1]!=-1:
                            precision_correct += 1
                            break 

                total_true += len(true)
                    
            precision = precision_correct/total_pred
            recall = precision_correct/total_true
            f1 = 2*precision*recall/(precision+recall)
            print ("Doc exact: \n precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
            return precision, recall, f1



## token wise F1
# def doc_partial_match(pred_file, truth_file):
#     with open(pred_file, 'r') as fp:
#         with open(truth_file, 'r') as ft:
#             pred_list = fp.readlines()
#             true_list = ft.readlines()
#             assert len(pred_list) == len(true_list), "Test cases mismatch"
#             total_n = len(pred_list)  ## total number of docs
#             total_pred = 0  ## total number of preds
#             correct = 0

#             for i in range(total_n):
#                 true = true_list[i].strip().split('|')
#                 true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

#                 preds = pred_list[i].strip().split('|')
#                 preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

#                 for j in range(len(preds)):
#                     total_pred += 1
#                     for t in true:
#                         t_start = t[0]
#                         t_end = t[1]
#                         if (preds[j][0]<=t_start and preds[j][1]>=t_start+(t_end-t_start)//2) \
#                             or (preds[j][0]<=t_start+(t_end-t_start)//2 and preds[j][1]>=t_end) \
#                             or (preds[j][0]>=t_start and preds[j][1]<=t_end and (preds[j][1]-preds[j][0])>=(t_end-t_start)//2):
#                             correct += 1
#                             break

#             return correct / total_pred

def doc_partial_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_n = len(pred_list)  ## total number of docs
            total_pred = 0
            total_true = 0
            precision_correct = 0

            for i in range(total_n):
                true = true_list[i].strip().split('|')
                true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for p in preds:
                    pred = list(range(p[0], p[1]+1))
                    if -1 in pred:
                        continue
                    total_pred += (len(pred))
                    for t in true:
                        t = list(range(t[0], t[1]+1))
                        if -1 not in t:
                            precision_correct += len(set(pred) & set(t))

                for t in true:
                    t = list(range(t[0], t[1]+1))
                    total_true += (len(t))
                   

            precision = precision_correct / total_pred
            recall = precision_correct / total_true 
            f1 = 2*precision*recall / (precision+recall)
            print ("Doc Token wise: \n precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
            return precision, recall, f1        



def seg_exact_match(pred_list, true_list, pred_out_dir, gold_dir):
    pred_list = [[int(a) for a in x] for x in pred_list]
    true_list = [[int(a) for a in x] for x in true_list]

    # pred_lines = []
    # true_lines = []

    pred_f = open(pred_out_dir, 'w+') 
    true_f = open(gold_dir, 'w+')

    for i in range(len(pred_list)):
        first = 0
        j = 0
        string = ''
        no_mention = True
        while j<len(pred_list[i]):
            while j<len(pred_list[i]) and pred_list[i][j]== 0:
                j+=1
            if j<len(pred_list[i]) and pred_list[i][j] == 1:
                no_mention=False
                start = j
                while j+1<len(pred_list[i]) and pred_list[i][j+1]==1:
                    j+=1
                end = j 
                if first > 0:
                    string += " | "
                string += (str(start)+' '+str(end))
                j+=1
                first += 1
        if no_mention:
            pred_f.write("-1 -1"+'\n')
        else:
            pred_f.write(string+'\n')

        # pred_lines.append(string)


    for i in range(len(true_list)):
        first = 0
        j = 0
        string = ''
        no_mention = True
        while j<len(true_list[i]):
            while j<len(true_list[i]) and true_list[i][j]== 0:
                j+=1
            if j<len(true_list[i]) and true_list[i][j] == 1:
                no_mention=False
                start = j
                while j+1<len(true_list[i]) and true_list[i][j+1]==1:
                    j+=1
                end = j 
                if first > 0:
                    string += " | "
                string += (str(start)+' '+str(end))
                j+=1
                first += 1
        if no_mention:
            true_f.write("-1 -1"+'\n')
        else:
            true_f.write(string+'\n')
            
        # true_lines.append(string)

    # assert len(pred_lines) == len(true_lines), "Test cases mismatch"
    # total_n = len(pred_lines)  ## number of docs
    # total_pred = 0  ## number of preds
    # total_true = 0  ## number of truth
    # precision_correct = 0 ## pred in truth

    # for i in range(total_n):
    #     true = true_lines[i].strip().split('|')
    #     true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

    #     preds = pred_lines[i].strip().split('|')
    #     preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

    #     total_pred += len(preds)
    #     for p in range(len(preds)):
    #         for t in true:
    #             if preds[p][0]==t[0] and preds[p][1]==t[1]:
    #                 precision_correct += 1
    #                 break 

    #     total_true += len(true)

    pred_f.close()
    true_f.close()
            
    # precision = precision_correct/total_pred
    # recall = precision_correct/total_true
    # f1 = 2*precision*recall/(precision+recall)
    # print ("Segment Exact mention match: \n precision: {}, recall: {}, f1: {}".format(precision, recall, f1))
    # return precision, recall, f1

    p, r, f = doc_exact_match(pred_out_dir, gold_dir)
    return p, r, f




'''
TODO:
evaluation: 
token-wise P, R, F1
mention-wise P, R, F1

'''












