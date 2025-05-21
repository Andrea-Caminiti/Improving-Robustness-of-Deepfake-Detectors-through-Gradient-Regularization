# Improving Robustness of Deepfake Detectors through Gradient Regularization

Project for the Computer Vision class at Sapienza University of Rome a.y. 2024-2025

Dataset to be downloaded from: [here](https://drive.google.com/file/d/1d1T8XpJcvee6GxS5VnwlCKsRkSfNmvh9/view?usp=sharing)
## Introduction
Aim of the project is to test the hypothesis that the gradient regularization loss presented in [[1]](https://ieeexplore.ieee.org/document/10516609)
may be helpful in making DeepFake Detection models more robust to adversarial attacks. 

### The Dataset 

The dataset used is a downsamples version of the DFFD dataset from [[2]](https://arxiv.org/pdf/1910.01717). As shown in the 'stats.txt' file, found in the Data folder, the dataset is made up of 70,000 images, 35,000 real images and 35,000 deepfake images. The real images have been extracted from the CelebA part of the DFFD dataset while the deepfakes have been randomly extracted from the multiple deepfake categories. 

## The model

The model chosen for the task is an EfficientNet-b0 architecture whose Squeeze-and-Excitation (SE) Blocks have been swapped with Convolutional Block Attention Modules (CBAM).

Two models have been trained, one with the gradient regularization loss of [1], one with The Binary Cross Entropy loss.

## The experiments
Both model have been trained for 50 epochs, with an Adam optimizer and a learning rate of 1e-4.

Also a simple pipeline has been used to load the dataset in chuncks to comply with the hardware constraints. 

Early stopping is used to stop if the validation loss is not decreasing fast enough (threshold of 1e-3 decrease). 

The best models are then used for testing and robustness to attacks. In both cases, the model after the last epoch is used. Three type of attacks have been chosen:     
- Two Whitebox attacks (FGSM [[3]](https://arxiv.org/abs/1412.6572v3), PGD [[4]](https://arxiv.org/abs/1706.06083))
- One Blackbox attack (Random Noise Patch [[5]](https://arxiv.org/abs/1712.09665))

## Results

Result are summarized in the tables below, the adversarial metrics are the mean of five different seeds (3233, 42, 0, 1234, 2025).


|Model| Accuracy | Precision | Recall | F1-Score | AUC |
|-----|-----|-----|-----|-----|-----|
Simple | 99.96% | 99.96% | 99.96% | 99.96% | 99.96% | 99
Gradient Regularized | 99.98% | 99.98% | 99.98% | 99.98% | 99.98% |

|Model |Attack | Accuracy | Precision | Recall | F1-Score  | AUC  |
|-----|-----|-----|-----|-----|-----|-----|
Simple | FGSM | 99.69% | 99.69% | 99.69% | 99.69% | 99.69% |
Gradient Regularized | FGSM | 98.75% | 98.75% | 98.75% | 98.75% | 98.75% |
|-----|-----|-----|-----|-----|-----|-----|
Simple | PGD | 72.66% | 72.66% | 72.66% | 72.66% | 72.66% |
Gradient Regularized | PGD | 74.54% | 74.54% |74.54% |74.54% |74.54% |
|-----|-----|-----|-----|-----|-----|-----|
Simple | Patch | 81.68% | 81.68% | 81.68% | 81.68% | 81.68% |
Gradient Regularized | Patch | 82.96% | 82.96% | 82.96% | 82.96% | 82.96% |

It's easy to see that out of three attacks only in two gradient regularization helped in making the model more robust, even though not by a lot. The Wilcoxon signed-rank test applied to the results of all five different seed per attack had the p values summarized in the last table:

|Attack| P-value |
|-----|-----|
## References

[1] W. Guan, W. Wang, J. Dong and B. Peng, "Improving Generalization of Deepfake Detectors by Imposing Gradient Regularization," in IEEE Transactions on Information Forensics and Security

[2] On the Detection of Digital Face Manipulation Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil
Jain, (2020), In: Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2020), Seattle,
WA, Jun. 2020

[3] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,”

[4]Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks.

[5] Adversarial Patch. Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer