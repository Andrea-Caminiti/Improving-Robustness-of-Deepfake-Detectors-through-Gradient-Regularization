import torch 
import torch.nn as nn
from tqdm import tqdm 
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score 
from Network.model import CBAMEfficientNet
from Train.loss import GradientRegularizedLoss

def train(dLoader_train, dLoader_valid, epochs, save_path, log_path, patience, loss_fn = nn.BCEWithLogitsLoss(), device = 'cuda'):
    model = CBAMEfficientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(True))
    gradientRegularized = isinstance(loss_fn, GradientRegularizedLoss)
    losses = {'Train': [], 'Valid': []}
    metrics = {'Train': {'accuracy': [], 'precision': [], 'recall': [], 'f-score': [], 'auc': []},
               'Valid': {'accuracy': [], 'precision': [], 'recall': [], 'f-score': [], 'auc': []}}
    with open(log_path, 'w') as log:
        for epoch in range(epochs):
            model.train()
            epoch_loss = []        
            preds, gt = [], []
            best_valid_loss = 0.0
            for batch in tqdm(dLoader_train, desc=f'Epoch {epoch}: Training...', total=len(dLoader_train)):
                optimizer.zero_grad()

                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits, shallow_feat = model(imgs)

                if gradientRegularized:
                    loss = loss_fn(model, logits, shallow_feat, labels)
                else: 
                    loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                preds.append(logits.argmax(dim=1).cpu())
                gt.append(labels.cpu())

            losses['Train'].append(np.mean(epoch_loss).item())
            metrics['Train']['accuracy'].append(accuracy_score(gt, preds))
            prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
            auc = roc_auc_score(gt, preds)
            metrics['Train']['precision'].append(prec)
            metrics['Train']['recall'].append(rec)
            metrics['Train']['f-score'].append(f)
            metrics['Train']['auc'].append(auc)
            l, acc, prec, rec, f, auc = valid(model, dLoader_valid, epoch, device)
            losses['Valid'].append(l)
            metrics['Valid']['accuracy'].append(acc)
            metrics['Valid']['precision'].append(prec)
            metrics['Valid']['recall'].append(rec)
            metrics['Valid']['f-score'].append(f)
            metrics['Valid']['auc'].append(auc)

            if best_valid_loss == 0.0 or l < best_valid_loss:
                #Save the most accurate model 
                best_valid_loss = l
                if not os.path.exists(r'./models'):
                    os.mkdir(r'./models')
                torch.save(model, rf'./models/{"Gradient Regularized" if gradientRegularized else "Baseline"} at epoch {epoch}.pt')
                
            if l > best_valid_loss and patience_c == patience:
                    break
                    
            elif l > best_valid_loss:
                patience_c += 1
            
            else: 
                patience_c = 0

    losses = pd.DataFrame(losses)
    metrics_t = pd.DataFrame(metrics['Train'])
    metrics_v = pd.DataFrame(metrics['Valid'])
    losses.to_csv('logs/Losses.csv')
    metrics_t.to_csv('logs/Training Metrics.csv')
    metrics_v.to_csv('logs/Validation Metrics.csv')

    return
    
def valid(model, dLoader_valid, epoch, device):
    loss_fn = nn.BCEWithLogitsLoss()
    epoch_loss = []        
    preds, gt = [], []
    with torch.no_grad():
        for batch in tqdm(dLoader_valid, desc=f'Epoch {epoch}: Validation...', total=len(dLoader_valid)):

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

    return loss, acc, prec, rec, f, auc
