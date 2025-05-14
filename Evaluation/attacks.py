import torchattacks
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader 
import torch
from tqdm import tqdm
import numpy as np
from Data.dataset import create_dataset

def attacks(model, test_dpath, device, log_path):
    loss_fn = nn.BCEWithLogitsLoss()
    datas_loss = []        
    preds, gt = [], []
    attack_dict ={
                'FGSM': torchattacks.FGSM(model, eps=8/255),
                'PGD': torchattacks.PGD(model, steps=40),
                'CW':  torchattacks.CW(model, steps=1000),
                'PATCH': torchattacks.Square(model),
                'ONE PIXEL': torchattacks.OnePixel(model, pixels=5, steps=75, popsize=400)
                }
    for name, attack in attack_dict.items():
        for dataset in create_dataset(test_dpath):
            batch_loss = []
            datas_prec = []
            datas_rec = []
            datas_f = []
            datas_auc = []
            datas_acc = []
            dLoader_test = DataLoader(dataset, 32, False)
            with torch.no_grad():
                for batch in tqdm(dLoader_test, desc=f'Testing...', total=len(dLoader_test)):

                    imgs, labels = batch
                    imgs = imgs.to(device)
                    labels = F.one_hot(labels.long(), 2).float().to(device)
                    adv_imgs = attack(imgs, labels)

                    logits, shallow_feat = model(adv_imgs)

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
    

        with open(log_path + f' {name} attack.txt', 'w') as log:
            log.write('#'*75 + '\n' + f'Evaluation of attack: {name} results'.center(75) + '\n' + '#'*75 + '\n')
            log.write(f'Loss: {loss}\n')
            log.write(f'Accuracy: {acc}\n')
            log.write(f'Precision: {prec}\n')
            log.write(f'Recall: {rec}\n')
            log.write(f'F1-score: {f}\n')
            log.write(f'Area Under the Curve: {auc}\n')


    return