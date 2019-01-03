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
def doc_exact_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_n = len(pred_list)  ## total number of docs
            total_pred = 0  ## total number of preds
            correct = 0

            for i in range(total_n):
                true = true_list[i].strip().split('|')
                true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for p in range(len(preds)):
                    total_pred += 1
                    for t in true:
                        if preds[p][0]==t[0] and preds[p][1]==t[1]:
                            correct += 1
                            break

            return correct / total_pred




def doc_partial_match(pred_file, truth_file):
    with open(pred_file, 'r') as fp:
        with open(truth_file, 'r') as ft:
            pred_list = fp.readlines()
            true_list = ft.readlines()
            assert len(pred_list) == len(true_list), "Test cases mismatch"
            total_n = len(pred_list)  ## total number of docs
            total_pred = 0  ## total number of preds
            correct = 0

            for i in range(total_n):
                true = true_list[i].strip().split('|')
                true = [(int(t.split()[0]), int(t.split()[1])) for t in true]

                preds = pred_list[i].strip().split('|')
                preds = [(int(p.split()[0]), int(p.split()[1])) for p in preds]

                for j in range(len(preds)):
                    total_pred += 1
                    for t in true:
                        t_start = t[0]
                        t_end = t[1]
                        if (preds[j][0]<=t_start and preds[j][1]>=t_start+(t_end-t_start)//2) \
                            or (preds[j][0]<=t_start+(t_end-t_start)//2 and preds[j][1]>=t_end) \
                            or (preds[j][0]>=t_start and preds[j][1]<=t_end and (preds[j][1]-preds[j][0])>=(t_end-t_start)//2):
                            correct += 1
                            break

            return correct / total_pred
















