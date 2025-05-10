import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score 
from Data.dataset import create_dataset

def test(model, test_dpath, device, log_path):
    loss_fn = nn.BCEWithLogitsLoss()
    datas_loss = []        
    preds, gt = [], []
    for dataset in create_dataset(test_dpath):
        batch_loss = []
        datas_prec = []
        datas_rec = []
        datas_f = []
        datas_auc = []
        datas_acc = []
        dLoader_test = torch.utils.data.DataLoader(dataset, 32, False)
        with torch.no_grad():
            for batch in tqdm(dLoader_test, desc=f'Testing...', total=len(dLoader_test)):

                imgs, labels = batch
                imgs = imgs.to(device)
                labels = F.one_hot(labels.long(), 2).float().to(device)

                logits, shallow_feat = model(imgs)

                loss = loss_fn(logits, labels)

                batch_loss.append(loss.item())
                preds.append(logits.argmax(dim=1).cpu())
                gt.append(labels.cpu())
            datas_loss.append(np.mean(batch_loss).item())
            datas_acc.append(accuracy_score(gt, preds))
            prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
            datas_prec.append(prec)
            datas_rec.append(rec)
            datas_f.append(f)
            datas_auc.append(roc_auc_score(gt, preds))
        
        loss = np.mean(datas_loss).item()
        acc = np.mean(datas_acc).item()
        prec = np.mean(datas_prec).item()
        rec = np.mean(datas_rec).item()
        f = np.mean(datas_f).item()
        auc = np.mean(datas_auc).item()
    

    with open(log_path, 'w') as log:
        log.write('#'*75 + '\n' + 'Evaluation results'.center(75) + '\n' + '#'*75 + '\n')
        log.write(f'Test Loss: {loss}\n')
        log.write(f'Test accuracy: {acc}\n')
        log.write(f'Test precision: {prec}\n')
        log.write(f'Test recall: {rec}\n')
        log.write(f'Test F1-score: {f}\n')
        log.write(f'Test Area Under the Curve: {auc}\n')


    return

def plots(csv_dir):

    path = os.path.join(csv_dir, 'Losses.csv')
    loss = pd.read_csv(path, sep=',')

    path = os.path.join(csv_dir, 'Training Metrics.csv')
    metrics_t = pd.read_csv(path, sep=',')

    path = os.path.join(csv_dir, 'Validation Metrics.csv')
    metrics_v = pd.read_csv(path, sep=',')

    fig, axes = plt.subplots(1, 6)
    fig.set_figwidth(24)
    fig.set_figheight(5)

    axes[0].set_title('Losses per epoch')
    axes[0].plot(loss['Train'], label='Training Loss')
    axes[0].plot(loss['Valid'], label='Validation Loss')
    axes[0].legend(loc='upper left')

    axes[0].set_title('Accuracies per epoch')
    axes[0].plot(metrics_t['accuracy'], label='Training Accuracy')
    axes[0].plot(metrics_v['accuracy'], label='Validation Accuracy')
    axes[0].legend(loc='lower left')

    axes[0].set_title('Precisions per epoch')
    axes[0].plot(loss['precision'], label='Training Precision')
    axes[0].plot(loss['precision'], label='Validation Precision')
    axes[0].legend(loc='lower left')

    axes[0].set_title('Recalls per epoch')
    axes[0].plot(loss['recall'], label='Training Recall')
    axes[0].plot(loss['recall'], label='Validation Recall')
    axes[0].legend(loc='lower left')

    axes[0].set_title('F1-scores per epoch')
    axes[0].plot(loss['f-score'], label='Training F1-Score')
    axes[0].plot(loss['f-score'], label='Validation F1-Score')
    axes[0].legend(loc='lower left')

    axes[0].set_title('Area Under the Curve per epoch')
    axes[0].plot(loss['auc'], label='Training Area Under the Curve')
    axes[0].plot(loss['auc'], label='Validation Area Under the Curve')
    axes[0].legend(loc='lower left')
        
