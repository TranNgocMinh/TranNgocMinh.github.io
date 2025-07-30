import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


class SelfContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(SelfContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, image_embeds, text_embeds, neutral_embeds):
        pos_sim = F.cosine_similarity(image_embeds, text_embeds)
        neg_sim = F.cosine_similarity(image_embeds, neutral_embeds)
        loss = F.relu(self.margin + neg_sim - pos_sim).mean()
        return loss


class SelfAttentionRegularization(nn.Module):
    def __init__(self):
        super(SelfAttentionRegularization, self).__init__()

    def forward(self, attention_map):
        return torch.norm(attention_map, p=2)


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, reconstructed, target):
        return self.loss_fn(reconstructed, target)


class TotalLoss(nn.Module):
    def __init__(self, weights, lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.3):
        super(TotalLoss, self).__init__()
        self.ce = WeightedCrossEntropyLoss(weights)
        self.contrast = SelfContrastiveLoss()
        self.recon = ReconstructionLoss()
        self.attn = SelfAttentionRegularization()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

    def forward(self, logits, labels, image_embeds, text_embeds, neutral_embeds, recon_img, target_img, attention_map):
        ce_loss = self.ce(logits, labels)
        contrast_loss = self.contrast(image_embeds, text_embeds, neutral_embeds)
        recon_loss = self.recon(recon_img, target_img)
        attn_loss = self.attn(attention_map)
        return (self.lambda1 * ce_loss +
                self.lambda2 * contrast_loss +
                self.lambda3 * recon_loss +
                self.lambda4 * attn_loss)
