# Improving Robustness of Deepfake Detectors through Gradient Regularization

Project for the Computer Vision class at Sapienza University of Rome a.y. 2024-2025

Dataset to be downloaded from: [here](https://drive.google.com/file/d/11j4Ion6h0CFb7Vx5AOLiVEU-JUQCYVXj/view?usp=sharing)
## Introduction
Aim of the project is to test the hypothesis that the gradient regularization loss presented in [[1]](https://ieeexplore.ieee.org/document/10516609)
may be helpful in making DeepFake Detection models more robust to adversarial attacks. 

### The Dataset 

The dataset used is a downsamples version of the DFFD dataset from [[2]](https://arxiv.org/pdf/1910.01717). As shown in the `stats.txt` file, found in the `Data` folder, the dataset is made up of 185,000 images, 100,000 real images and 85.000 deepfake images. The real images have been extracted from the `CelebA` and `ffhq` part of the DFFD dataset while the deepfakes have been randomly extracted from the multiple deepfake categories. 

Examples of the samples: 
![](figs/Real%20Samples.png)
![](figs/Fake%20Samples.png)

## The model

The model chosen for the task is an EfficientNet-b0 architecture whose Squeeze-and-Excitation (SE) Blocks have been swapped with Convolutional Block Attention Modules (CBAM).

Two models have been trained, one with the gradient regularization loss of [1], one with The Binary Cross Entropy loss.

## The experiments
Both model have been trained for 20 epochs, with an Adam optimizer and a learning rate of 1e-4.

Also a simple pipeline has been used to load the dataset in chuncks to comply with the hardware constraints. 

Early stopping is used to stop if the validation loss is not decreasing fast enough (threshold of 1e-1 decrease). 

The best models are then used for testing and robustness to attacks. In both cases, the model after the last epoch is used. Three type of attacks have been chosen:     
- Two Whitebox attacks (FGSM [[3]](https://arxiv.org/abs/1412.6572v3), PGD [[4]](https://arxiv.org/abs/1706.06083))
- One Blackbox attack (Random Noise Patch [[5]](https://arxiv.org/abs/1712.09665))

Examples of images out of the attack are:
![](figs/Adversarial%20Images/Baseline%20FGSM.png)
![](figs/Adversarial%20Images/Regularized%20PGD.png)
![](figs/Adversarial%20Images/Baseline%20PATCH.png)



## Results

Result are summarized in the tables below, the adversarial metrics are the mean of five different seeds (3233, 42, 0, 1234, 2025).

### Baseline losses and metrics over 20 epochs
![](figs/Baseline%20losses%20and%20metrics.png)


### Gradient Regularized model losses and metrics over 20 epochs
![](figs/Gradient%20Regularized%20losses%20and%20metrics.png)

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
|FGDS| 0.0625 |
|PGD|0.0625 |
|Patch|0.0625 |

Thus we reject the alternative hypothesis and there is no statistical significance in the difference in metrics for neither of the three attacks.

Other experiments have been carried out with more epochs (50) to see if a more trained model would be more impacted by the attacks and if in that case the regularization helped to make the model more robust, but still no improvements has been reached. All output files are found in folder `Results 2`
## References

[1] W. Guan, W. Wang, J. Dong and B. Peng, "Improving Generalization of Deepfake Detectors by Imposing Gradient Regularization," in IEEE Transactions on Information Forensics and Security

[2] On the Detection of Digital Face Manipulation Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, Anil
Jain, (2020), In: Proceedings of IEEE Computer Vision and Pattern Recognition (CVPR 2020), Seattle,
WA, Jun. 2020

[3] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,”

[4]Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks.

[5] Adversarial Patch. Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer