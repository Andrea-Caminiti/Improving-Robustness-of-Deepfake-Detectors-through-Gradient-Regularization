import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from Data.dataset import create_dataset

def test(model, test_dpath, device, log_path):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    datas_loss = []    
    datas_prec = []
    datas_rec = []
    datas_f = []
    datas_auc = []
    datas_acc = []    
    global_preds, gloabal_gt = [], []
    for dataset in create_dataset(test_dpath):
        batch_loss = []
        preds, gt = [], []
        dLoader_test = torch.utils.data.DataLoader(dataset, 32, False)
        with torch.no_grad():
            for batch in tqdm(dLoader_test, desc=f'Testing...', total=len(dLoader_test), leave=False):

                imgs, labels = batch
                imgs = imgs.to(device)
                lab = labels
                labels = F.one_hot(labels.long(), 2).float().to(device)

                logits, shallow_feat = model(imgs)

                loss = loss_fn(logits, labels)

                batch_loss.append(loss.item())
                preds += logits.argmax(dim=1).cpu().tolist()
                gt += lab
            datas_loss.append(np.mean(batch_loss).item())
            datas_acc.append(accuracy_score(gt, preds))
            prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
            datas_prec.append(prec)
            datas_rec.append(rec)
            datas_f.append(f)
            print(np.unique(gt))
            datas_auc.append(roc_auc_score(gt, preds))
            gloabal_gt += gt
            global_preds += preds
    
    loss = np.mean(datas_loss).item()
    acc = np.mean(datas_acc).item()
    prec = np.mean(datas_prec).item()
    rec = np.mean(datas_rec).item()
    f = np.mean(datas_f).item()
    auc = np.mean(datas_auc).item()
    confusion = confusion_matrix(gloabal_gt, global_preds)

    with open(log_path, 'w') as log:
        log.write('#'*75 + '\n' + 'Evaluation results'.center(75) + '\n' + '#'*75 + '\n')
        log.write(f'Test Loss: {loss}\n')
        log.write(f'Test accuracy: {acc}\n')
        log.write(f'Test precision: {prec}\n')
        log.write(f'Test recall: {rec}\n')
        log.write(f'Test F1-score: {f}\n')
        log.write(f'Test Area Under the Curve: {auc}\n')
    
    sb.heatmap(confusion, annot=True, cmap="Blues", fmt='.0f', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f'{log_path.split("/")[-1][:-4]}' + ' Confusion Matrix')
    if not os.path.exists('./figs'):
        os.mkdir('./figs')
    plt.savefig(rf'./figs/{log_path.split("/")[-1][:-4]} confusion matrix.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()


    return

def plots(csv_dir):

    path = os.path.join(csv_dir, f'{csv_dir.split("/")[-1]} Losses.csv')
    loss = pd.read_csv(path, sep=',')

    path = os.path.join(csv_dir, f'{csv_dir.split("/")[-1]} Training Metrics.csv')
    metrics_t = pd.read_csv(path, sep=',')

    path = os.path.join(csv_dir, f'{csv_dir.split("/")[-1]} Validation Metrics.csv')
    metrics_v = pd.read_csv(path, sep=',')

    fig = plt.figure(figsize=(20, 20), layout="constrained")
    spec = fig.add_gridspec(4, 2)

    ax0 = fig.add_subplot(spec[0, :])
    ax10 = fig.add_subplot(spec[1, 0])
    ax11 = fig.add_subplot(spec[1, 1])

    ax20 = fig.add_subplot(spec[2, 0])
    ax21 = fig.add_subplot(spec[2, 1])
    
    ax3 = fig.add_subplot(spec[3, :])
    
    axes = [[ax0], [ax10, ax11], [ax20, ax21], [ax3]]

    axes[0][0].set_title('Losses per epoch')
    axes[0][0].plot(np.arange(0, len(loss['Train']), 1), loss['Train'], label='Training Loss')
    axes[0][0].plot(np.arange(0, len(loss['Train']), 1), loss['Valid'], label='Validation Loss')
    axes[0][0].legend(loc='upper right')

    axes[1][0].set_title('Accuracies per epoch')
    axes[1][0].plot(np.arange(0, len(loss['Train']), 1), metrics_t['accuracy'], label='Training Accuracy')
    axes[1][0].plot(np.arange(0, len(loss['Train']), 1), metrics_v['accuracy'], label='Validation Accuracy')
    axes[1][0].legend(loc='lower right')

    axes[1][1].set_title('Precisions per epoch')
    axes[1][1].plot(np.arange(0, len(loss['Train']), 1), metrics_t['precision'], label='Training Precision')
    axes[1][1].plot(np.arange(0, len(loss['Train']), 1), metrics_v['precision'], label='Validation Precision')
    axes[1][1].legend(loc='lower right')

    axes[2][0].set_title('Recalls per epoch')
    axes[2][0].plot(np.arange(0, len(loss['Train']), 1), metrics_t['recall'], label='Training Recall')
    axes[2][0].plot(np.arange(0, len(loss['Train']), 1), metrics_v['recall'], label='Validation Recall')
    axes[2][0].legend(loc='lower right')

    axes[2][1].set_title('F1-scores per epoch')
    axes[2][1].plot(np.arange(0, len(loss['Train']), 1), metrics_t['f-score'], label='Training F1-Score')
    axes[2][1].plot(np.arange(0, len(loss['Train']), 1), metrics_v['f-score'], label='Validation F1-Score')
    axes[2][1].legend(loc='lower right')

    axes[3][0].set_title('Area Under the Curve per epoch')
    axes[3][0].plot(np.arange(0, len(loss['Train']), 1), metrics_t['auc'], label='Training Area Under the Curve')
    axes[3][0].plot(np.arange(0, len(loss['Train']), 1), metrics_v['auc'], label='Validation Area Under the Curve')
    axes[3][0].legend(loc='lower right')
    
    if not os.path.exists('./figs'):
        os.mkdir('./figs')
    plt.savefig(rf'./figs/{csv_dir.split("/")[-1]} losses and metrics.png')
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
        
