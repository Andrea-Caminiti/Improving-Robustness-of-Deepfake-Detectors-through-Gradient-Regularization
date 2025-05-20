import torchattacks
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader 
import torch
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sb
import os
from Data.dataset import create_dataset

def apply_patch(images, labels, patch_size=32, value=1.0):
    
    B, C, H, W = images.size()
    patched_images = images.clone()

    for i in range(B):
       
        x_start = np.random.randint(0, H - patch_size)
        y_start = np.random.randint(0, W - patch_size)

        patch = torch.rand((C, patch_size, patch_size)).to(images.device)
        patched_images[i, :, x_start:x_start+patch_size, y_start:y_start+patch_size] = patch

    return patched_images

def attacks(model, test_dpath, device, log_path):
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    datas_loss = []      
    datas_prec = []
    datas_rec = []
    datas_f = []
    datas_auc = []
    datas_acc = []  
    global_preds, global_gt = [], []
    attack_dict ={
                'FGSM': torchattacks.FGSM(model, eps=8/255),
                'PGD': torchattacks.PGD(model, steps=40),
                'CW':  torchattacks.CW(model, steps=1000),
                'PATCH': apply_patch,
                }
    for name, attack in attack_dict.items():
        for dataset in tqdm(create_dataset(test_dpath), desc=f'Attack: {name}...'):
            batch_loss = []
            preds, gt = [], []
            dLoader_test = DataLoader(dataset, 8, False)
            for batch in tqdm(dLoader_test, desc=f'Testing...', total=len(dLoader_test), leave=False):
                
                imgs, labels = batch
                lab = labels.long()
                labels = F.one_hot(labels.long(), 2).float().to(device)
                adv_imgs = attack(imgs, lab).to(device)
                logits = model(adv_imgs)

                loss = loss_fn(logits, labels)

                batch_loss.append(loss.item())
                preds += logits.argmax(dim=1).cpu().tolist()
                gt += lab
                
                del logits, lab, labels, adv_imgs, imgs
                gc.collect()
            datas_loss.append(np.mean(batch_loss).item())
            datas_acc.append(accuracy_score(gt, preds))
            prec, rec, f, _ = precision_recall_fscore_support(gt, preds, average='macro')
            datas_prec.append(prec)
            datas_rec.append(rec)
            datas_f.append(f)
            datas_auc.append(roc_auc_score(gt, preds))
            global_gt += gt
            global_preds += preds
            
            break
            
        loss = np.mean(datas_loss).item()
        acc = np.mean(datas_acc).item()
        prec = np.mean(datas_prec).item()
        rec = np.mean(datas_rec).item()
        f = np.mean(datas_f).item()
        auc = np.mean(datas_auc).item()
        confusion = confusion_matrix(global_gt, global_preds)


        with open(log_path + f' {name} attack.txt', 'w') as log:
            log.write('#'*75 + '\n' + f'Evaluation of attack: {name} results'.center(75) + '\n' + '#'*75 + '\n')
            log.write(f'Loss: {loss}\n')
            log.write(f'Accuracy: {acc}\n')
            log.write(f'Precision: {prec}\n')
            log.write(f'Recall: {rec}\n')
            log.write(f'F1-score: {f}\n')
            log.write(f'Area Under the Curve: {auc}\n')
            
        sb.heatmap(confusion, annot=True, cmap="Blues", fmt='.0f')
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.title(f'{log_path.split("/")[-1][:-4]} {name}' + ' Confusion Matrix')
        if not os.path.exists('./figs'):
            os.mkdir('./figs')
        plt.savefig(rf'./figs/{log_path.split("/")[-1][:-4]} {name} confusion matrix.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    
    return