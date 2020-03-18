import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

class MetricsCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, valid_data, title):
        super(MetricsCallback, self).__init__()
        self.title = title
        self.X_valid = valid_data[0]
        self.Y_valid = valid_data[1]
        self.losses = []
        self.accuracy = []
        self.val_loss = []
        self.val_acc = []
        self.val_bad_recall = []
        self.w_precision = []
        self.w_recall = []
        self.w_f1 = []
        
    def specify_recall(self, cm, index):
        tp = cm[index][index]
        tp_fn = sum(cm[index])
        return tp/tp_fn
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch:
            self.losses.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))
            self.val_loss.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))
            
            Y_pred = self.model.predict_classes(self.X_valid).ravel()
            cm = confusion_matrix(self.Y_valid, Y_pred)
            br = self.specify_recall(cm, 0)
            self.val_bad_recall.append(br)
            
            p, r, f1, _ = precision_recall_fscore_support(self.Y_valid, Y_pred, average='weighted')
            self.w_precision.append(p)
            self.w_recall.append(r)
            self.w_f1.append(f1)
        if epoch % 10 == 0:
            print(".", end=" ")
            self.model.save_weights("models/"+self.title+'/'+str(epoch)+'/my_model')

    
    def plot(self, name):
        iters = range(len(self.losses))
        plt.figure()
        
        # acc
        plt.plot(iters, self.accuracy, 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses, 'g', label='train loss')
        # val_acc
        plt.plot(iters, self.val_acc, 'b', label='val acc')
        # val_bad_recall
        plt.plot(iters, self.val_bad_recall, 'c', label='val bad recall')
        # val_weight_F1
        plt.plot(iters, self.w_f1, '#B088FF', label='val weight F1')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('metrics')
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.02,0.8),loc='center left')
        screenshot_path = "screenshot"
        if not os.path.isdir(screenshot_path): os.mkdir(screenshot_path)
        plt.savefig(screenshot_path+"/"+self.title+"_"+str(name)+".png")
        plt.show()
        