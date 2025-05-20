import torch
import torch.nn as nn
from torch.autograd import grad

class PIM(nn.Module):
    def __init__(self, r, device):
        super().__init__()
        self.r = r
        self.delta_mean = nn.Parameter(torch.zeros((1,24,1,1))).to(device) # 24 being the number of channels out of the shallow part of the model
        self.delta_std = nn.Parameter(torch.zeros((1,24,1,1))).to(device)
        
    def forward(self, x, grad_mean=None, grad_std=None):
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        std = x.std(dim=(0, 2, 3), keepdim=True)
        
        if grad_mean != None and grad_std != None:
            norm = torch.sqrt((grad_mean ** 2 + grad_std ** 2).sum()) + 1e-8
            self.delta_mean.data = self.r * grad_mean / norm
            self.delta_std.data = self.r * grad_std / norm
        
        x_norm = (x-mean) / (std + 1e-8)
        mean_p = mean + self.delta_mean
        std_p = std + self.delta_std
        
        return x_norm * std_p + mean_p

class GradientRegularizedLoss(nn.Module):
    def __init__(self,  r, alpha, criterion = nn.BCEWithLogitsLoss(), device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.criterion = criterion
        self.pim = PIM(r, device=device)
    
    def forward(self, model, shallow_feat, labels):
        
        pim_feat = self.pim(shallow_feat)
        pim_logits = model.classifier(model.deep(pim_feat))
        loss = self.criterion(pim_logits, labels)

        grads_mean = grad(loss, self.pim.delta_mean, create_graph=True, retain_graph=True)[0]
        grads_std  = grad(loss, self.pim.delta_std, create_graph=True, retain_graph=True)[0]
        # Inject perturbations to feature stats
        pim_feat = self.pim(shallow_feat, grads_mean, grads_std)
        pim_logits = model.classifier(model.deep(pim_feat))
        
        loss_perturbed = self.criterion(pim_logits, labels)

        # Combined loss
        total_loss = (1 - self.alpha) * loss + self.alpha * loss_perturbed

        return total_loss

# import torch
# import torch.nn as nn
# from torch.autograd import grad

# class GradientRegularizedLoss(nn.Module):
#     def __init__(self,  r, alpha, criterion = nn.BCEWithLogitsLoss()):
#         super().__init__()
#         self.r = r
#         self.alpha = alpha
#         self.criterion = criterion
    
#     def forward(self, model, shallow_feat, labels):

#         # Loss on perturbed output
#         B, C, *_ = shallow_feat.shape
#         grad_mean = torch.zeros((B,C,1,1)).to(shallow_feat.device)
#         grad_std = torch.zeros((B,C,1,1)).to(shallow_feat.device)
#         grad_mean.requires_grad = True
#         grad_std.requires_grad = True
        
#         shallow_feat_perturbed = self.perturb_feature_stats(shallow_feat, grad_mean, grad_std, self.r)
#         deep_feat = model.deep(shallow_feat)
#         logits = model.classifier(deep_feat)
#         loss = self.criterion(logits, labels)
#         # Compute gradients w.r.t. shallow feature stats        
#         grad_mean = grad(loss, grad_mean, create_graph=True, retain_graph=True)[0]
#         grad_std  = grad(loss, grad_std, create_graph=True, retain_graph=True)[0]

#         # Inject perturbations to feature stats
#         shallow_feat_perturbed = self.perturb_feature_stats(shallow_feat, grad_mean, grad_std, self.r)

#         # Forward pass with perturbed features
#         deep_feat = model.deep(shallow_feat_perturbed)
#         logits_perturbed = model.classifier(deep_feat.view(deep_feat.size(0), -1))

#         # Loss on perturbed output
#         loss_perturbed = self.criterion(logits_perturbed, labels)

#         # Combined loss
#         total_loss = (1 - self.alpha) * loss + self.alpha * loss_perturbed

#         return total_loss
    
#     def perturb_feature_stats(self, features, grad_mean, grad_std, r):
#         # Normalize features
#         B, C, H, W = features.size()
#         mean = features.mean(dim=[2,3], keepdim=True)
#         std = features.std(dim=[2,3], keepdim=True)

#         normed_feat = (features - mean) / (std + 1e-8)

#         # Perturb mean and std along their gradient directions
#         delta_mean = r * grad_mean / (grad_mean.norm(p=2) + 1e-8)
#         delta_std  = r * grad_std / (grad_std.norm(p=2) + 1e-8)

#         mean_perturbed = mean + delta_mean.view(B, C, 1, 1)
#         std_perturbed  = std + delta_std.view(B, C, 1, 1)

#         # Re-normalize features with perturbed stats
#         feat_perturbed = normed_feat * (std_perturbed + 1e-8) + mean_perturbed
#         return feat_perturbed
