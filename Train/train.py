import torch 
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import pandas as pd
import os
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score 
from Network.model import CBAMEfficientNet
from Train.loss import GradientRegularizedLoss
from Data.dataset import create_dataset

def train(train_dpath, valid_dpath, epochs, patience, loss_fn = nn.BCEWithLogitsLoss(), device = 'cuda'):
    model = CBAMEfficientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(True), lr= 1e-4)
    gradientRegularized = isinstance(loss_fn, GradientRegularizedLoss)
    losses = {'Train': [], 'Valid': []}
    metrics = {'Train': {'accuracy': [], 'precision': [], 'recall': [], 'f-score': [], 'auc': []},
               'Valid': {'accuracy': [], 'precision': [], 'recall': [], 'f-score': [], 'auc': []}}
    
    for epoch in tqdm(range(epochs), desc='Training...'):
        model = model.train()
        datas_loss = []        
        preds, gt = [], []
        best_valid_loss = 0.0
        i = 0
        for dataset in create_dataset(train_dpath):
            batch_loss = []
            datas_prec = []
            datas_rec = []
            datas_f = []
            datas_auc = []
            datas_acc = []
            i+=1
            dLoader_train = torch.utils.data.DataLoader(dataset, 32, True)
            for batch in tqdm(dLoader_train, desc=f'Epoch {epoch}: Dataset {i}...', total=len(dLoader_train), leave=False):
                optimizer.zero_grad()
                imgs, labels = batch
                imgs = imgs.to(device)
                lab = labels
                labels = F.one_hot(labels.long(), 2).float().to(device)

                logits, shallow_feat = model(imgs)

                if gradientRegularized:
                    loss = loss_fn(model, logits, shallow_feat, labels)
                else: 
                    loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                preds += logits.argmax(dim=1).cpu().tolist()
                gt += lab
            del dLoader_train
            gc.collect()
            prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
            datas_loss.append(np.mean(batch_loss).item())
            datas_prec.append(prec)
            datas_rec.append(rec)
            datas_f.append(f)
            datas_auc.append(roc_auc_score(gt, preds))
            datas_acc.append(accuracy_score(gt, preds))

        losses['Train'].append(np.mean(datas_loss).item())
        metrics['Train']['accuracy'].append(np.mean(datas_acc).item())
        
        metrics['Train']['precision'].append(np.mean(datas_prec).item())
        metrics['Train']['recall'].append(np.mean(datas_rec).item())
        metrics['Train']['f-score'].append(np.mean(datas_f).item())
        metrics['Train']['auc'].append(np.mean(datas_auc).item())

        valid_l, valid_acc, valid_prec, valid_rec, valid_f, valid_auc = [], [], [], [], [], []
        for dataset in create_dataset(valid_dpath): 
            i+=1
            dLoader_valid = torch.utils.data.DataLoader(dataset, 32, False)
            l, acc, prec, rec, f, auc = valid(model, dLoader_valid, epoch, device)
            valid_l.append(l)
            valid_prec.append(prec)
            valid_rec.append(rec)
            valid_f.append(f)
            valid_auc.append(auc)
            valid_acc.append(acc)
        del dLoader_valid
        gc.collect()
        losses['Valid'].append(np.mean(valid_l).item())
        metrics['Train']['accuracy'].append(np.mean(valid_acc).item())
        
        metrics['Valid']['precision'].append(np.mean(valid_prec).item())
        metrics['Valid']['recall'].append(np.mean(valid_rec).item())
        metrics['Valid']['f-score'].append(np.mean(valid_f).item())
        metrics['Valid']['auc'].append(np.mean(valid_auc).item())
        l = np.mean(valid_l).item()
        if best_valid_loss == 0.0 or l < best_valid_loss:
            #Save the most accurate model 
            best_valid_loss = l
            if not os.path.exists(r'./models'):
                os.mkdir(r'./models')
            torch.save(model.state_dict(), rf'./models/{"Gradient Regularized" if gradientRegularized else "Baseline"} at epoch {epoch}.pt')
            
        if l > best_valid_loss and patience_c == patience:
                break
                
        elif l > best_valid_loss:
            patience_c += 1
        
        else: 
            patience_c = 0

    losses = pd.DataFrame(losses)
    metrics_t = pd.DataFrame(metrics['Train'])
    metrics_v = pd.DataFrame(metrics['Valid'])
    if not os.path.exists('logs'):
        os.makedirs('logs')
    losses.to_csv('logs/Losses.csv')
    metrics_t.to_csv('logs/Training Metrics.csv')
    metrics_v.to_csv('logs/Validation Metrics.csv')

    return
    
def valid(model, dLoader_valid, epoch, device):
    loss_fn = nn.BCEWithLogitsLoss()
    epoch_loss = []        
    preds, gt = [], []
    model = model.eval()
    with torch.no_grad():
        for batch in tqdm(dLoader_valid, desc=f'Epoch {epoch}: Validation...', total=len(dLoader_valid)):

            imgs, labels = batch
            imgs = imgs.to(device)
            lab = labels
            labels = F.one_hot(labels.long(), 2).float().to(device)

            logits, _ = model(imgs)

            loss = loss_fn(logits, labels)

            epoch_loss.append(loss.item())
            preds += logits.argmax(dim=1).cpu().tolist()
            gt += lab
        loss = np.mean(epoch_loss).item()
        acc = accuracy_score(gt, preds)
        prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
        auc = roc_auc_score(gt, preds)

    return loss, acc, prec, rec, f, auc
