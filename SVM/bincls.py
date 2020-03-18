class BinaryClassificationAverageReport():
    # Only for Binary Classification
    def __init__(self, target_names):
        self.accum_cm = []
        self.target_names = target_names
        
    def cm_append(self, cm):
        # postive -> bad , negative -> good
        tp, fp, fn, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        self.accum_cm.append([tp, fp, fn, tn])
        
    def object_score(self):
        fold_num = len(self.accum_cm)
        avg_positive_precision = 0
        avg_negative_precision = 0
        avg_positive_recall = 0
        avg_negative_recall = 0
        avg_positive_F1 = 0
        avg_negative_F1 = 0
        avg_weight_F1 = 0
        avg_acc = 0
    
        for cm in self.accum_cm:
            positive_total = cm[0] + cm[1]
            negative_total = cm[2] + cm[3]
            total = positive_total + negative_total
            positive_pred = cm[0] + cm[2]
            negative_pred = cm[1] + cm[3]
            pp, pr, np, nr = (cm[0]/positive_pred), (cm[0]/positive_total), (cm[3]/negative_pred), (cm[3]/negative_total)
            avg_positive_precision += pp
            avg_positive_recall += pr
            avg_negative_precision += np 
            avg_negative_recall += nr
            positive_F1 = ((2*pp*pr)/(pp+pr))
            negative_F1 = ((2*np*nr)/(np+nr))
            avg_positive_F1 += positive_F1
            avg_negative_F1 += negative_F1
            avg_weight_F1 += ((positive_total/total)*positive_F1+(negative_total/total)*negative_F1)
            avg_acc += ((cm[0]+cm[3])/total)
        
        avg_positive_precision = avg_positive_precision/fold_num
        avg_positive_recall = avg_positive_recall/fold_num
        avg_negative_precision = avg_negative_precision/fold_num
        avg_negative_recall = avg_negative_recall/fold_num
        avg_positive_F1 = avg_positive_F1/fold_num
        avg_negative_F1 = avg_negative_F1/fold_num
        avg_weight_F1 = avg_weight_F1/fold_num
        avg_acc = avg_acc/fold_num
        
        return 0.6 * avg_positive_recall + 0.4 * avg_weight_F1

    def avg_cm_report(self):
        fold_num = len(self.accum_cm)
        avg_positive_precision = 0
        avg_negative_precision = 0
        avg_positive_recall = 0
        avg_negative_recall = 0
        avg_positive_F1 = 0
        avg_negative_F1 = 0
        avg_weight_F1 = 0
        avg_acc = 0
    
        for cm in self.accum_cm:
            positive_total = cm[0] + cm[1]
            negative_total = cm[2] + cm[3]
            total = positive_total + negative_total
            positive_pred = cm[0] + cm[2]
            negative_pred = cm[1] + cm[3]
            pp, pr, np, nr = (cm[0]/positive_pred), (cm[0]/positive_total), (cm[3]/negative_pred), (cm[3]/negative_total)
            avg_positive_precision += pp
            avg_positive_recall += pr
            avg_negative_precision += np 
            avg_negative_recall += nr
            positive_F1 = ((2*pp*pr)/(pp+pr))
            negative_F1 = ((2*np*nr)/(np+nr))
            avg_positive_F1 += positive_F1
            avg_negative_F1 += negative_F1
            avg_weight_F1 += ((positive_total/total)*positive_F1+(negative_total/total)*negative_F1)
            avg_acc += ((cm[0]+cm[3])/total)
    
        avg_positive_precision = avg_positive_precision/fold_num
        avg_positive_recall = avg_positive_recall/fold_num
        avg_negative_precision = avg_negative_precision/fold_num
        avg_negative_recall = avg_negative_recall/fold_num
        avg_positive_F1 = avg_positive_F1/fold_num
        avg_negative_F1 = avg_negative_F1/fold_num
        avg_weight_F1 = avg_weight_F1/fold_num
        avg_acc = avg_acc/fold_num
        
        print()
        print()
        print("Below number are the average of %d fold." % (fold_num))
        print()
        print(self.target_names[0])
        print('%23s %8.2f%s' % ('precision:', avg_positive_precision*100,'%'))
        print('%23s %8.2f%s' % ('recall:', avg_positive_recall*100,'%'))
        print('%23s %8.2f%s' % ('F1:', avg_positive_F1*100,'%'))
        print(self.target_names[1])
        print('%23s %8.2f%s' % ('precision:', avg_negative_precision*100,'%'))
        print('%23s %8.2f%s' % ('recall:', avg_negative_recall*100,'%'))
        print('%23s %8.2f%s' % ('F1:', avg_negative_F1*100,'%'))
        print("---------------------------------")
        print('%23s %8.2f%s' % ('weight_F1:', avg_weight_F1*100,'%'))
        print('%23s %8.2f%s' % ('acc:', avg_acc*100,'%'))
        print()
        
