import torch
import torch.nn as nn


## ~~~~~~~~~~~~ Projected Distribution ~~~~~~~~~~~~
class PDLoss(nn.Module):
    def __init__(self, target_feature, device='cpu'):
        super(PDLoss, self).__init__()
        features = PDLoss._vectorize_features(target_feature).detach()
        self.target, _ = features.sort(dim=0)
        self.device = device

    def forward(self, input):
        features = PDLoss._vectorize_features(input)
        features, _ = features.sort(dim=0)
        features_losses = torch.sum(torch.abs(features - self.target), dim=0)
        # self.loss = features_losses.mean()
        self.loss = features_losses.sum()
        return input

    @staticmethod
    def _vectorize_features(features):
        batch_size, depth, c, d = features.size()
        # batch size(=1)
        # depth = number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        # resize exracted feautres into phi=R(nxm), n=hxw
        return features.view(batch_size * depth, c * d).T
