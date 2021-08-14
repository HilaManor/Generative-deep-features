"""

"""
import torch
import torch.nn as nn


# ~~~~~~~~~~~~ Projected Distribution ~~~~~~~~~~~~
class PDLoss(nn.Module):
    """

    """
    def __init__(self, target_feature, device='cpu'):
        """
        The function creates a new instance of PDLoss, based on the paper: "Projected Distribution
         Loss for Image Enhancement" by Mauricio Delbracio, Hossein Talebei and Peyman Milanfar.

        :param target_feature: The target feature for calculating loss - the loss will be calculated
         in respect to this value. (the "desired" feature).
         Must be sortable object.
        :param device: The device on which we use the loss.
        """
        super(PDLoss, self).__init__()
        features = PDLoss._vectorize_features(target_feature).detach()
        self.target, _ = features.sort(dim=0)
        self.device = device
        self.loss = None

    def forward(self, input_f):
        """
        Calculate the PDLoss between self.target and input_f, and updates self.loss
        with the result.
        :param input_f: The loss is calculated between input_f (input features) and
         self.target_features.
        :return: the function returns input_f (used for transferring the gradients).
        """
        features = PDLoss._vectorize_features(input_f)
        features, _ = features.sort(dim=0)
        features_losses = torch.mean(torch.abs(features - self.target), dim=0)
        self.loss = features_losses.mean()
        return input_f

    @staticmethod
    def _vectorize_features(features):
        """
        Reshape the 4D-input into 2D-output.
        :param features: The feature maps will be reshaped.
        :return: 2D reshape of the given input.
        """
        batch_size, depth, c, d = features.size()
        # batch size(=1)
        # depth = number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        # resize exracted feautres into phi=R(nxm), n=hxw
        return features.view(batch_size * depth, c * d).T
