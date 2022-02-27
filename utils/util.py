import dgl
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import functional as F

def plotROC(y, z, pstr=''):
    fpr, tpr, tt = metrics.roc_curve(y, z)
    roc_auc = roc_auc_score(y, z)
    plt.figure()
    plt.plot(fpr, tpr, 'o-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.title('ROC ' + pstr + ' AUC: '+str(roc_auc_score(y, z)))


def evaluate_metrics(y, y_pred, y_proba, draw_roc=False):

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    ba = balanced_accuracy_score(y, y_pred)
    tpr = recall_score(y, y_pred)
    tnr = tn/(tn+fp)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    if draw_roc:
        plotROC(y, y_pred)
    
    return tn, fp, fn, tp, round(ba, 3), round(tpr, 3), round(tnr, 3), round(f1, 3), round(auc, 3)


def print_metrics(y_proba, y_actual):
    
    # arr_len = len(y_proba_arr)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = 0, 0, 0, 0, 0, 0

    for threshold in thresholds:
        total_ba, total_tpr, total_tnr, total_f1, total_mcc, total_auc = 0, 0, 0, 0, 0, 0

        y_pred_list = (np.array(y_proba) >= threshold).astype(int)
        tn, fp, fn, tp, ba, tpr, tnr, f1, auc = evaluate_metrics(y_actual, y_pred_list, y_proba)
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        total_ba += ba
        total_tpr += tpr
        total_tnr += tnr
        total_f1 += f1
        total_mcc += mcc
        total_auc += auc

        if total_ba > best_ba:
            best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = total_ba, total_tpr, total_tnr, total_f1, total_mcc, total_auc

    print("============== Average Performance =================\n",
          '* BA :', "{:.3f}".format(best_ba), '\n',
          '* TPR :', "{:.3f}".format(best_tpr), '\n',
          '* TNR :', "{:.3f}".format(best_tnr), '\n',
          '* F1-score :', "{:.3f}".format(best_f1), '\n',
          '* MCC :', "{:.3f}".format(best_mcc), '\n',
          '* AUC :', "{:.3f}".format(best_auc), '\n',
          '======================================================')


def predict(args, model, data_loader, criterion, optimizer, is_train):

    device = args['DEVICE']

    total_loss, correct = 0, 0
    output_total, y_total, y_pred_total = [], [], []

    for i, X_data, y_data in data_loader:
        if args['MODEL'] == 'acgcn-mmp':
            smiles1 = [x[0]['GRAPH_SMILES1'] for x in X_data]
            smiles2 = [x[0]['GRAPH_SMILES2'] for x in X_data]
            y_data = torch.from_numpy(np.array(y_data)).float()

            batch_smiles1 = dgl.batch(smiles1)
            batch_smiles2 = dgl.batch(smiles2)

            if torch.cuda.is_available():
                batch_smiles1 = batch_smiles1.to(device)
                batch_smiles2 = batch_smiles2.to(device)
                y_data = y_data.to(device)

            outputs = model(batch_smiles1, batch_smiles2)

        elif args['MODEL'] == 'acgcn-sub':
            core = [x[0]['GRAPH_CORE'] for x in X_data]
            sub1 = [x[0]['GRAPH_SUB1'] for x in X_data]
            sub2 = [x[0]['GRAPH_SUB2'] for x in X_data]
            y_data = torch.from_numpy(np.array(y_data)).float()

            batch_core = dgl.batch(core)
            batch_sub1 = dgl.batch(sub1)
            batch_sub2 = dgl.batch(sub2)

            if torch.cuda.is_available():
                batch_core = batch_core.to(device)
                batch_sub1 = batch_sub1.to(device)
                batch_sub2 = batch_sub2.to(device)
                y_data = y_data.to(device)

            outputs = model(batch_core, batch_sub1, batch_sub2)

        loss = criterion(outputs, y_data)
        output_total += outputs.tolist()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        y_pred = (outputs >= 0.5).float()
        correct += (y_pred == y_data).float().sum()
        y_total += y_data.tolist()
        y_pred_total += [int(i) for i in y_pred]

    bal_acc = balanced_accuracy_score(y_total, y_pred_total)

    return model, loss, total_loss, bal_acc, output_total


def get_actual_label(data_loader):
    
    y_arr = []
    for i, X_data, y_data in data_loader:
        y_arr += y_data.tolist()
    
    return y_arr


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        self.eps = 1e-9

    def forward(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * torch.log(output + self.eps)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output + self.eps))
        else:
            loss = target * torch.log(output + self.eps) + (1 - target) * torch.log(1 - output + self.eps)
            print(output, target)
            print(loss)
        return torch.neg(torch.mean(loss))
