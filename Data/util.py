import os
from PIL import Image
import cv2
import shutil
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm
from dataset import Dataset, BaseTransform, Augmentation

def rename(dirPath):
    for f in os.listdir(dirPath):
        os.rename(os.path.join(dirPath, f), os.path.join(dirPath, f'R_ffhq_{f}'))

def convert(dirPath):
    for f in os.listdir(dirPath):
        im = Image.open(os.path.join(dirPath, f))
        im.save(os.path.join(dirPath, f"{f[:-4]}.png"))
        os.remove(os.path.join(dirPath, f))

def count(dirPath):
    c = {'R': 0, 'F': 0}
    for elem in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, elem)):
            c['R'] += elem[0] == 'R'
            c['F'] += elem[0] == 'F'
        else:
            d = count(os.path.join(dirPath, elem))
            c['R'] += d['R']
            c['F'] += d['F']
    return c

def downsample(dirPath, i, j):
    for elem in tqdm(os.listdir(dirPath), desc='Sorting the dataset...'):
        
        if os.path.isfile(os.path.join(dirPath, elem)) and i < 100_000 and elem[0] == 'R':
            if i < 70_000:
                idx_i = (i%14) + 1
                if not os.path.exists(f'CV Dataset\Train\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Train\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Train\Real\{idx_i}', elem))
            elif i < 80_000: 
                idx_i = (i%2) + 1
                if not os.path.exists(f'CV Dataset\Validation\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Validation\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(fr'CV Dataset\Validation\Real\{idx_i}', elem))
            else: 
                idx_i = (i%4) + 1
                if not os.path.exists(f'CV Dataset\Test\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Test\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Test\Real\{idx_i}', elem))
            i+=1
        
        if os.path.isfile(os.path.join(dirPath, elem)) and j < 100_000 and elem[0] == 'F':
            if j < 70_000:
                idx_j = (j%14) + 1
                if not os.path.exists(f'CV Dataset\Train\Fake\{idx_j}'):
                    os.makedirs(f'CV Dataset\Train\Fake\{idx_j}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Train\Fake\{idx_j}', elem))
            elif j < 80_000: 
                idx_j = (j%2) + 1
                if not os.path.exists(f'CV Dataset\Validation\Fake\{idx_j}'):
                    os.makedirs(f'CV Dataset\Validation\Fake\{idx_j}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(fr'CV Dataset\Validation\Fake\{idx_j}', elem))
            else: 
                idx_j = (j%4) + 1
                if not os.path.exists(f'CV Dataset\Test\Fake\{idx_j}'):
                    os.makedirs(f'CV Dataset\Test\Fake\{idx_j}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Test\Fake\{idx_j}', elem))
            j+=1

    return i, j

def get_shapes(dirPath):
    shapes = set()
    for elem in os.listdir(dirPath):
        if os.path.isdir(os.path.join(dirPath, elem)):
            shapes = shapes.union(get_shapes(os.path.join(dirPath, elem)))
            
        else:
            size = Image.open(os.path.join(dirPath, elem)).size
            shapes.add(size)

    return shapes

def resize(dirPath):
    
    for elem in os.listdir(dirPath):
        if os.path.isdir(os.path.join(dirPath, elem)):
            resize(os.path.join(dirPath, elem))
        else:
            im = Image.open(os.path.join(dirPath, elem))
            if im.size != (299, 299):
                im = im.resize((299, 299), resample=Image.Resampling.NEAREST)
                im.save(os.path.join(dirPath, elem))

    return

def load(dirPath):
    imgs, labels = [], []
    for d in tqdm(os.listdir(dirPath), desc=f'Loading {dirPath}...', leave=False):
        if os.path.isdir(os.path.join(dirPath, d)):
            ims, labs = load(os.path.join(dirPath, d))
            imgs += ims
            labels += labs
        else:
            im = cv2.imread(os.path.join(dirPath, d))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            imgs.append(im)
            if d[0] == 'F':
                labels.append(1)
            else:
                labels.append(0)

    return imgs, labels

def show_samples(data_path, title="Sample"):
    
    imgs, labels = load(data_path)

    ims = random.choices(imgs, k=5)

    fig, axes = plt.subplots(1, 5, figsize=(12, 4))
    for i in range(5):
        axes[i].imshow(ims[i])

    plt.suptitle(title, weight="bold")
    plt.tight_layout()
    plt.savefig(f"figs/{title}.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    
def wilcoxon_test(baseline, regulazied):
    stat, p_value = wilcoxon(baseline, regulazied)
    print('P_value: ', p_value)

def save_to_npy(data_path):
    tf = BaseTransform()
    aug = Augmentation(prob = 0.5)
    for i in range(len(os.listdir(os.path.join(data_path, 'Real')))):
            real_im, real_lab = load(os.path.join(data_path, f'Real\{i+1}'))
            fake_im, fake_lab = load(os.path.join(data_path, f'Fake\{i+1}'))
            datas = Dataset(real_im + fake_im, real_lab + fake_lab, tf, aug, mode = 'Train' if 'Train' in data_path else 'Eval')
            datas.shuffle()
            if not os.path.exists(os.path.join(data_path, f'{i+1}')):
                os.makedirs(os.path.join(data_path, f'{i+1}'))
            np.save(os.path.join(data_path, f'{i+1}/images.npy'), datas.imgs)
            np.save(os.path.join(data_path, f'{i+1}/labels.npy'), datas.labels)

if __name__ == '__main__':
    dirPath = r'CV Dataset'
    train_dataPath = 'CV Dataset\Train'
    valid_dataPath = 'CV Dataset\Validation'
    test_dataPath = 'CV Dataset\Test'
    save_to_npy(train_dataPath)
    save_to_npy(valid_dataPath)
    save_to_npy(test_dataPath)
    with open('Data/stats.txt', 'a') as f:
        # f.write(f'Before downsampling we have {c["R"]} real images and {c["F"]} fake ones\n')
        # downsample(dirPath, 0, 0)
        # c = count(dirPath)
        # f.write(f'After downsampling and moving we have {c["R"]} real images and {c["F"]} fake ones\n')
        #f.write(f'Images have shapes: {get_shapes(dirPath)}\n')
        # resize(dirPath)
        # f.write(f'After resizing the images have shapes: {get_shapes(dirPath)}\n')
        #show_samples('CV Dataset\Validation\Real', title="Real Samples")
        #show_samples('CV Dataset\Validation\Fake', title="Fake Samples")
        pass

    # baseline_FGSM = np.array([0.9935,  0.996, 0.997, 0.996,  0.999])
    # regularized_FGSM = np.array([0.987, 0.986, 0.989, 0.989, 0.985])
    # print('FGSM')
    # wilcoxon_test(baseline_FGSM, regularized_FGSM)
    
    # baseline_PGD = np.array([0.726, 0.729, 0.727, 0.726, 0.723])
    # regularized_PGD = np.array([0.745, 0.743, 0.745, 0.749, 0.745])
    # print('PGD')
    # wilcoxon_test(baseline_PGD, regularized_PGD)
    
    # baseline_patch = np.array([0.819, 0.819, 0.816, 0.815, 0.814])
    # regularized_patch = np.array([0.832, 0.826, 0.828, 0.832, 0.828])
    # print('PATCH')
    # wilcoxon_test(baseline_patch, regularized_patch)