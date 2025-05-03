import os
from PIL import Image

def rename(dirPath):
    for f in os.listdir(dirPath):
        os.rename(os.path.join(dirPath, f), os.path.join(dirPath, f'R_celeba_{f}'))

def convert(dirPath):
    for f in os.listdir(dirPath):
        im = Image.open(os.path.join(dirPath, f))
        im.save(os.path.join(dirPath, f"{f[:-4]}.png"))
        os.remove(os.path.join(dirPath, f))


if __name__ == '__main__':
    dirPath = 'CV Dataset\CelebA\img_align_celeba'
    convert(dirPath)