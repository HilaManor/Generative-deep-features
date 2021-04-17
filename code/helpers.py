import torch
from torchvision import transforms, models

import matplotlib.pyplot as plt
from PIL import Image

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
    
unloader = transforms.ToPILImage()  # reconvert into PIL image

def image_loader(image_name, loader=loader):
    image = Image.open(image_name) #.convert('RGB')
    #image = image.filter(ImageFilter.GaussianBlur(GAUSSIAN_BLUR))
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, unloader=unloader, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, title, unloader=unloader):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(title)