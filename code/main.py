from __future__ import print_function

## ~~~~~~~~~~~~~~~~~~~ Imports ~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models
import copy
import time
import helpers
import net_model

## ~~~~~~~~~~~~~~~~~ Environment ~~~~~~~~~~~~~~~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# del cnn
torch.cuda.empty_cache()
cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

style_img = helpers.image_loader("../images/starry_night_full.jpg",device)
#content_img = image_loader("./images/.jpg")

# assert style_img.size() == content_img.size(), \
#     "we need to import style and content images of the same size"

plt.ion()

plt.figure()
helpers.imshow(style_img, title='Style Image')

# plt.figure()
# helpers.imshow(content_img, title='Content Image')

# input_img = content_img.clone()
# if you want to use white noise instead uncomment the below line:
base_img = torch.randn(style_img.data.size(), device=device)
#base_img = torch.poisson(base_img) #torch.zeros_like(style_img, device=device) #
print(style_img.shape)
#base_img[0,:,0:250,:] = 1
# add the original input image to the figure:
plt.figure()
helpers.imshow(base_img, title='Input Image')


input_img = copy.deepcopy(base_img)

STYLE_LAYERS = ['conv1_1', 'conv2_1']#, 'conv3_1', 'conv4_1', 'conv5_1']#, 'conv_3', 'conv_4', 'conv_5']
# STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']
#STYLE_WEIGHTS = np.array([1,0.75,0.2,0.2,0.2])
STYLE_WEIGHTS = np.array([1,1,1,1,1])
# STYLE_WEIGHTS = np.ones(len(STYLE_LAYERS))
EPOCHS = 1000
start_time = time.time()
output, losses = net_model.run_style_transfer(device,cnn, cnn_normalization_mean, cnn_normalization_std,
#                             content_img, 
                            style_img, input_img,  
                            STYLE_LAYERS, STYLE_WEIGHTS,
                            num_steps=EPOCHS)
end_time = time.time()
print(f"The training process took {end_time-start_time}")

plt.figure()
helpers.imshow(output, title='Output Image')

fig = plt.figure(figsize=(16,10))
fig.add_subplot(2,2,1)
plt.plot(losses) 
plt.title('Losses vs Iter')
fig.add_subplot(2,2,2)
plt.plot(np.log10(losses)) 
plt.title('log10(Losses) vs Iter')
fig.add_subplot(2,2,3)
plt.plot(np.arange(len(losses))[-50:], losses[-50:]) 
plt.title('Zoomed Losses vs Iter')
fig.add_subplot(2,2,4)
plt.plot(np.arange(len(losses))[-50:], np.log10(losses[-50:])) 
plt.title('Zoomed log10(Losses) vs Iter')

helpers.imsave(output, f'outputs/sn_contxLossv1_{STYLE_LAYERS[-1]}_e{EPOCHS}_w{STYLE_WEIGHTS}.png')
fig.savefig(f'outputs/sn_contxLossv1_{STYLE_LAYERS[-1]}_e{EPOCHS}_w{STYLE_WEIGHTS}_fig.png')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()


