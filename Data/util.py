import os
from PIL import Image
import cv2
import shutil
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def rename(dirPath):
    for f in os.listdir(dirPath):
        os.rename(os.path.join(dirPath, f), os.path.join(dirPath, f'R_celeba_{f}'))

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
                idx_i = (i%70) + 1
                if not os.path.exists(f'CV Dataset\Train\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Train\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Train\Real\{idx_i}', elem))
            elif i < 80_000: 
                idx_i = (i%10) + 1
                if not os.path.exists(f'CV Dataset\Validation\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Validation\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(fr'CV Dataset\Validation\Real\{idx_i}', elem))
            else: 
                idx_i = (i%20) + 1
                if not os.path.exists(f'CV Dataset\Test\Real\{idx_i}'):
                    os.makedirs(f'CV Dataset\Test\Real\{idx_i}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Test\Real\{idx_i}', elem))
            i+=1
        
        if os.path.isfile(os.path.join(dirPath, elem)) and j < 100_000 and elem[0] == 'F':
            if j < 70_000:
                idx_j = (j%70) + 1
                if not os.path.exists(f'CV Dataset\Train\Fake\{idx_j}'):
                    os.makedirs(f'CV Dataset\Train\Fake\{idx_j}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(f'CV Dataset\Train\Fake\{idx_j}', elem))
            elif j < 80_000: 
                idx_j = (j%10) + 1
                if not os.path.exists(f'CV Dataset\Validation\Fake\{idx_j}'):
                    os.makedirs(f'CV Dataset\Validation\Fake\{idx_j}')
                shutil.move(os.path.join(dirPath, elem), os.path.join(fr'CV Dataset\Validation\Fake\{idx_j}', elem))
            else: 
                idx_j = (j%20) + 1
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
    

if __name__ == '__main__':
    dirPath = 'CV Dataset'
    # c = count(dirPath)
    # print(c)
    with open('Data/stats.txt', 'a') as f:
        # f.write(f'Before downsampling we have {c["R"]} real images and {c["F"]} fake ones\n')
        downsample(dirPath, 0, 0)
        # c = count(dirPath)
        # f.write(f'After downsampling and moving we have {c["R"]} real images and {c["F"]} fake ones\n')
        #f.write(f'Images have shapes: {get_shapes(dirPath)}\n')
        # resize(dirPath)
        # f.write(f'After resizing the images have shapes: {get_shapes(dirPath)}\n')
        #show_samples('CV Dataset\Validation\Real', title="Real Samples")
        #show_samples('CV Dataset\Validation\Fake', title="Fake Samples")