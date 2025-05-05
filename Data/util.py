import os
from PIL import Image
import shutil

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
    for elem in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, elem)) and i < 100_000 and elem[0] == 'R':
            if i < 70_000:
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Real\Train', elem))
            elif i < 80_000: 
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Real\Validation', elem))
            else: 
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Real\Test', elem))
            i+=1

        if os.path.isfile(os.path.join(dirPath, elem)) and j < 100_000 and elem[0] == 'F':
            if j < 70_000:
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Fake\Train', elem))
            elif j < 80_000: 
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Fake\Validation', elem))
            else: 
                shutil.move(os.path.join(dirPath, elem), os.path.join('CV Dataset\Fake\Test', elem))
            j+=1

        elif os.path.isdir(os.path.join(dirPath, elem)) and ('Fake' not in dirPath and 'Real' not in dirPath):
            h, k = downsample(os.path.join(dirPath, elem), i, j)
            i += h
            j += k
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

if __name__ == '__main__':
    dirPath = 'CV Dataset'
    # c = count(dirPath)
    # print(c)
    with open('Data/stats.txt', 'a') as f:
        # f.write(f'Before downsampling we have {c["R"]} real images and {c["F"]} fake ones\n')
        # downsample(dirPath, 0, 0)
        # c = count(dirPath)
        # f.write(f'After downsampling and moving we have {c["R"]} real images and {c["F"]} fake ones\n')
        #f.write(f'Images have shapes: {get_shapes(dirPath)}\n')
        resize(dirPath)
        f.write(f'After resizing the images have shapes: {get_shapes(dirPath)}\n')