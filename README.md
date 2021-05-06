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


## Insights
## 02/05/21
1. Rec Loss is affected by image size!
2. We achieve similar result w.r.t PDL style-transfer using CNN. 
3. When alpha is high, we don't get squares. (?)
4. Sanity check worked :) it depened on image size ( only at small scale, when we use fixed noise).
5. 
