import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    compare = pd.DataFrame(columns=('pred','true'))
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.reshape(labels.shape[0], 1).to(device)
        labels = labels.squeeze(1)
        
        optimizer.zero_grad()
        outputs, embed_temp = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
        probs = F.softmax(outputs, dim=1)
        probs = probs[:,1]
        _, predicteds = torch.max(outputs.data, 1)
            
        labels = labels.cpu()
        predicteds = predicteds.cpu()
        probs = probs.cpu()
        labels_list = np.array(labels).tolist()
        predicteds_list = np.array(predicteds).tolist()
        probs_list = np.array(probs.detach()).tolist()
        compare_temp = pd.DataFrame(columns=('pred','true'))
        compare_temp['true'] = labels_list
        compare_temp['pred'] = predicteds_list
        compare_temp['prob'] = probs_list
        compare = pd.concat([compare,compare_temp])

    compare_copy = compare.copy()
    accuracy, precision, recall, f1, roc_auc, prc_auc = metric(compare_copy)
    return total_loss/count, accuracy, f1, roc_auc, prc_auc


def metric(compare):
    y_true = compare['true']
    y_pred = compare['pred']
    y_true = y_true.astype('int64')
    y_pred = y_pred.astype('int64')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    y_prob = compare['prob']
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    prec, tpr, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    prc_auc = auc(tpr, prec)
    return accuracy, precision, recall, f1, roc_auc, prc_auc


def valid(model, device, valid_loader, criterion, epoch):
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.reshape(labels.shape[0], 1).to(device)
            labels = labels.squeeze(1)

            outputs, embed_temp = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            count += 1

            probs = F.softmax(outputs, dim=1)
            probs = probs[:,1]
            _, predicteds = torch.max(outputs.data, 1)
            
            labels = labels.cpu()
            predicteds = predicteds.cpu()
            probs = probs.cpu()
            labels_list = np.array(labels).tolist()
            predicteds_list = np.array(predicteds).tolist()
            probs_list = np.array(probs).tolist()
            compare_temp = pd.DataFrame(columns=('pred','true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = predicteds_list
            compare_temp['prob'] = probs_list
            compare = pd.concat([compare,compare_temp])
    compare_copy = compare.copy()
    accuracy, precision, recall, f1, roc_auc, prc_auc = metric(compare_copy)
    return total_loss/count, accuracy, f1, roc_auc, prc_auc


def save_model(current_acc, best_acc, log_dir, epoch, model, optimizer):
    is_best = current_acc > best_acc
    best_acc = max(current_acc, best_acc)
    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    log_dir_best = log_dir+'/Model_best.pth'
    if is_best:
        torch.save(checkpoint, log_dir_best)
    return best_acc
