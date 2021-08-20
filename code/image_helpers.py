from skimage import io as img
import torch
from torch import nn
from skimage import color
import numpy as np

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)


def read_image(image_path, im_channels_num, is_cuda):

    x = img.imread(image_path)
    x = np2torch(x, im_channels_num, is_cuda)
    x = x[:, 0:3, :, :]  # TODO WHY?
    return x


def np2torch(x, im_channels_num, is_cuda):
    if im_channels_num == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if is_cuda:
        x = x.to(torch.device('cuda'))  # todo do we need to reassign
    x = x.type(torch.cuda.FloatTensor) if is_cuda else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = inp[-1,:,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = inp[-1,-1,:,:].to(torch.device('cpu'))
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp