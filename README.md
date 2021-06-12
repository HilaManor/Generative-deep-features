<!-- [![Python 3.7.7](https://img.shields.io/badge/python-3.7.7+-blue.svg)](https://www.python.org/downloads/release/python-377/)
[![OpenCV](https://img.shields.io/badge/OpenCV-3.4.2-green)](https://opencv.org/) -->
<!--[![torch](https://img.shields.io/badge/torch-1.4.0-green)](https://pytorch.org/) -->
<!-- [![torchvision](https://img.shields.io/badge/torchvision-0.5.0-green)](https://pytorch.org/) -->

# Generative deep features


## Team
Hila Manor and Da-El Klang  
Supervised by Tamar Rott-Shaham

## Notes:
Q1 - How to init noise for image? The color channels are correlated...
Q2 - It seems that the weights of the style loss is always 1.

## GIT TOKEN:
how to set token:
git config --global credential.helper 'cache --timeout=3600'

## Insights
## 02/05/21
* **Image size matters**
  * ~~When alpha is high, we don't get squares~~
  * Probably the receptive field of the generator isn't big enough
  * Probably there are too little parameters
  * We achieved similar result w.r.t pixel-trained using CNN, and applying `alpha=0`
*  Rec Loss value is affected by image size! (needed alpha changes with image size)
* when introducting alpha, squares re-appear - with really big alpha we can get a reconstruction of the original image
  * But still get an original texture for random noise
  * When we made the alpha bigger (to get it to the same order of the style loss) the style loss also got bigger 😞
* ❓ Why is the noise in the first level fixed, not as stated in the paper?
## 28/05/21
It seems that high noise makes the fake image less grainy, but more blurry.
The the generator doesn't know how to generate small objects in the picture - maybe vgg min size issue?