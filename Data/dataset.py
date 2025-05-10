import torch
from torchvision import transforms
from Data.util import load 
import numpy as np
import os 

class BaseTransform:
    def __init__(self, size=(299, 299)):
        self.size = size

    def __call__(self, sample):
        # Convert to tensor
        if not isinstance(sample, torch.Tensor):
            image = transforms.ToTensor()(sample)
        # Normalize
        image = transforms.functional.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return image

class Augmentation:
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image):
        
        # 1. Random Horizontal Flip
        if np.random.rand() < self.prob:
            image = transforms.functional.hflip(image)

        # 2. Random Crop 
        if np.random.rand() < self.prob:
            # Get original dimensions
            h, w = image.shape[1:3]

            # Determine crop dimensions (between 80-100% of original dimensions)
            new_h = int(h * np.random.uniform(0.8, 1.0))
            new_w = int(w * np.random.uniform(0.8, 1.0))

            # Determine crop position
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            # Apply crop
            image = image[:, top:top + new_h, left:left + new_w]

            # Resize back to original dimensions
            image = transforms.functional.resize(image, (h, w),
                                              interpolation=transforms.InterpolationMode.NEAREST)

        # 3. Random Rotation (small angles)
        if np.random.rand() < self.prob:
            angle = np.random.uniform(-15, 15)  # Rotation of ±15 degrees
            image = transforms.functional.rotate(image, angle,
                                              interpolation=transforms.InterpolationMode.NEAREST)

        # 4. Random Color Jitter
        if np.random.rand() < self.prob:
            color_jitter = transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
            image = color_jitter(image)

        # 5. Random Gaussian Noise
        if np.random.rand() < self.prob:
            noise = torch.randn_like(image) * 0.02  # Adjust standard deviation to control noise intensity
            image = torch.clamp(image + noise, 0.0, 1.0)  # Clamp values between 0 and 1
        
        return image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transform = None, augmentation = None, mode = 'Train'):
        
        self.imgs, self.labels = np.array(imgs), np.array(labels)
        self.transform = transform
        self.augmentation = augmentation if mode == 'Train' else None

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        im = self.imgs[index]
        label = self.labels[index]

        if self.transform:
            im = self.transform(im)
        
        if self.augmentation:
            im = self.augmentation(im)

        return im, label
    
    def shuffle(self):
        idx = np.random.choice(np.arange(len(self.labels)), size=len(self.labels), replace=False)
        self.imgs = self.imgs[idx]
        self.labels = self.labels[idx]  
        


def create_dataset(data_path):
        
        tf = BaseTransform()
        aug = Augmentation(prob = 0.5)
        for i in range(len(os.listdir(os.path.join(data_path, 'Real')))):
            real_im, real_lab = load(os.path.join(data_path, f'Real\{i+1}'))
            fake_im, fake_lab = load(os.path.join(data_path, f'Fake\{i+1}'))
            datas = Dataset(real_im + fake_im, real_lab + fake_lab, tf, aug, mode = 'Train' if 'Train' in data_path else 'Eval')
            datas.shuffle()
            yield datas

if __name__ == '__main__':
    tf = BaseTransform()
    aug = Augmentation(prob = 0.7)
    d_set = Dataset(r'CV Dataset\Validation', tf, aug)
    