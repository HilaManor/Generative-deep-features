"""
Class for calculating Gram Loss.
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn

## ~~~~~~~~~~~~ Neural Style Transfer ~~~~~~~~~~~~
class GramLoss(nn.Module):
    """

    Forms gram loss object, based on style loss definition in the paper "Image Style Transfer Using
     Convolutional Neural Networks" by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge. The
    loss is MLE between thetarget and the input gram matrixes.

    """
    def __init__(self, target_feature,device='cpu'): #device for loss_model.generate_loss_block
        """
        Creates gram loss object, with given target feature.
        :param target_feature: The target feature for calculating the loss. Assumes 4D.
        """
        super(GramLoss, self).__init__()
        self.target = GramLoss._gram_matrix(target_feature).detach()

    def forward(self, input):
        """
        Calculates the loss of the given input feature in respect to the target feature, and updates
         self.loss accordingly.
        :param input: The loss is calculated between 4D-input (input feature map) and self.target.
        :return: input (used for transferring the gradients).
        """
        G = GramLoss._gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

    @staticmethod
    def _gram_matrix(input):
        """
        Calculate the gram matrix of the given input.
        :param input: 4D input data.
        :return: gram matrix of the input.
        """
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


