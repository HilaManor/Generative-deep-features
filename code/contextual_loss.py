import torch
import torch.nn as nn


class ContextualLoss(nn.Module):    
    def __init__(self, target_features, epsilon=1e-5, h=0.2):
        super(ContextualLoss, self).__init__()
        target = ContextualLoss._vectorize_features(target_features).detach()
        self.mu = torch.mean(target, 0)  # We assume feaure_vecotr = column
                                         # dim=1 for feature_vector=row
        self.target = target - self.mu  # centeralized features
        self.epsilon = epsilon
        self.h = h
        
    def forward(self, input):
        # {Y} = target features = style
        # {X} = source features = noise
        # y_j = one feature of the style image
        # x_i = one feature of the noise image
        
        features = ContextualLoss._vectorize_features(input)
        
        # if |X|!=|Y| sample N from the bigger set
        X_features = features - self.mu  # cenetralize the features
        N = self.target.shape[1]
        
#         CXs = torch.zeros(N, N)  # matrix containing in each row the CXij for that 
#                                  # feature i from the source image (compared to j 
#                                  # feature from target, which is in the column)
        CXs_max = torch.zeros(N)
        print(len(X_features.T))
        for idx, xi in enumerate(X_features.T):  # iterate over columns
            xi = xi.view(-1,1)
            
            di = xi.repeat(1, N).to(device='cpu')
            di = nn.functional.cosine_similarity(di, self.target.to(device='cpu'), dim=0)
#             di = nn.functional.cosine_similarity(di, self.target, dim=0)
            # SMALL d_ik = similar
            
            di_tilde = di / (di.min() + self.epsilon)
            del di
            # dij compared to the minimum d from THIS xi to ALL Y
            
            wi = torch.exp((1-di_tilde) / self.h)
            del di_tilde
            # d~_ij=1 -> w_ij=1 ; 
            # d~_ij > 1 -> 0< W_ij smaller < 1
            # d~_ij < 1 -> W_ij bigger > 1 <<<<<<<<<<<< want!
        
            Zi = wi.sum()
            CXi = wi / Zi
            del Zi
            del wi
#             CXs[idx,:] = CXi  
            CXs_max = torch.max(CXs_max, CXi.to(device='cpu'))
            del CXi
        
        most_similar_xs, _ = CXs_max.max(dim=0)
        CX = most_similar_xs.mean()  # most_similar_xs.sum() / len(most_similar_xs)
        
        self.loss = -torch.log(CX)      
        return input
    
    @staticmethod
    def _vectorize_features(features):
        batch_size, depth, c, d = features.size() 
        # batch size(=1)
        # depth = number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)
        
        # resise F_XL into \hat F_XL
        return features.view(batch_size * depth, c * d)[10:50] 
        
