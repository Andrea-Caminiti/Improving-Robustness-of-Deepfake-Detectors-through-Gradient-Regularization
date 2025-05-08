import torch
import torch.nn as nn
from torch.autograd import grad

class GradientRegularizedLoss(nn.Module):
    def __init__(self,  r, alpha, criterion = nn.BCEWithLogitsLoss()):
        self.r = r
        self.alpha = alpha
        self.criterion = criterion
    
    def forward(self, model, logits, shallow_feat, labels):
        
        loss = self.criterion(logits, labels)

        # Compute gradients w.r.t. shallow feature stats
        mean = shallow_feat.mean(dim=[2,3])
        std  = shallow_feat.std(dim=[2,3])

        grads_mean = grad(loss, mean, create_graph=True, retain_graph=True)[0]
        grads_std  = grad(loss, std, create_graph=True, retain_graph=True)[0]

        # Inject perturbations to feature stats
        shallow_feat_perturbed = self.perturb_feature_stats(shallow_feat, grads_mean, grads_std, self.r)

        # Forward pass with perturbed features
        deep_feat = model.deep(shallow_feat_perturbed)
        logits_perturbed = model.classifier(deep_feat.view(deep_feat.size(0), -1))

        # Loss on perturbed output
        loss_perturbed = self.criterion(logits_perturbed, labels)

        # Combined loss
        total_loss = (1 - self.alpha) * loss + self.alpha * loss_perturbed

        return total_loss
    
    def perturb_feature_stats(features, grads_mean, grads_std, r):
        # Normalize features
        B, C, H, W = features.size()
        mean = features.mean(dim=[2,3], keepdim=True)
        std = features.std(dim=[2,3], keepdim=True)

        normed_feat = (features - mean) / (std + 1e-6)

        # Perturb mean and std along their gradient directions
        delta_mean = r * grads_mean / (grads_mean.norm(p=2) + 1e-8)
        delta_std  = r * grads_std / (grads_std.norm(p=2) + 1e-8)

        mean_perturbed = mean + delta_mean.view(B, C, 1, 1)
        std_perturbed  = std + delta_std.view(B, C, 1, 1)

        # Re-normalize features with perturbed stats
        feat_perturbed = normed_feat * (std_perturbed + 1e-6) + mean_perturbed
        return feat_perturbed
