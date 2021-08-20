"""Class for calculating Gram Loss."""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn


# ~~~~~~~~~~~~ Neural Style Transfer ~~~~~~~~~~~~
class GramLoss(nn.Module):
    """Forms gram loss object, based on style loss definition in the paper "Image Style Transfer
    Using Convolutional Neural Networks" by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
    The loss is MLE between the target and the input gram matrixes."""
    def __init__(self, target_feature, device='cpu'):
        """Creates gram loss object, with given target feature.

        :param target_feature: The target feature for calculating the loss. Assumes 4D.
        :param device: device for loss_model.generate_loss_block
        """
        super().__init__()
        self.target = GramLoss._gram_matrix(target_feature).detach()
        self.device = device
        self.loss = None

    def forward(self, x):
        """Calculates the loss of the given input feature in respect to the target feature,
        and updates self.loss accordingly.

        :param x: The loss is calculated between 4D-input (input feature map) and self.target.
        :return: input (used for transferring the gradients).
        """
        g = GramLoss._gram_matrix(x)
        self.loss = nn.functional.mse_loss(g, self.target)
        return x

    @staticmethod
    def _gram_matrix(x):
        """Calculate the gram matrix of the given input.

        :param x: 4D input data.
        :return: gram matrix of the input.
        """
        a, b, c, d = x.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL

        g = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return g.div(a * b * c * d)
