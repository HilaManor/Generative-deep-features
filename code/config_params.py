import numpy as np

# desired size of the output image
#imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize = 512

# Optimizer to use. LBFGS/adam
OPTIMIZER = 'LBFGS'
lr = 1.3


STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']#, 'conv_3', 'conv_4', 'conv_5']
# STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8']


STYLE_WEIGHTS = np.array([1,0.75,0.2,0.2,0.2])
#STYLE_WEIGHTS = np.array([1,0.75,0.2,0.2,0.2])
#STYLE_WEIGHTS = np.array([1,1,1,1,1])
# STYLE_WEIGHTS = np.ones(len(STYLE_LAYERS))

EPOCHS = 4000

seed = 42

imshow_cycles = 250