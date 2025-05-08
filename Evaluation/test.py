import torch 
import torch.nn as nn
from tqdm import tqdm 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score 


def test(model, dLoader_test, device, log_path):
    loss_fn = nn.BCEWithLogitsLoss()
    epoch_loss = []        
    preds, gt = [], []
    with torch.no_grad():
        for batch in tqdm(dLoader_test, desc=f'Testing...', total=len(dLoader_test)):

            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits, shallow_feat = model(imgs)

            loss = loss_fn(logits, labels)

            epoch_loss.append(loss.item())
            preds.append(logits.argmax(dim=1).cpu())
            gt.append(labels.cpu())
        loss = np.mean(epoch_loss).item()
        acc = accuracy_score(gt, preds)
        prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
        auc = roc_auc_score(gt, preds)
    

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
        
