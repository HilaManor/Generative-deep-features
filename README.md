[![Python 3.8.5](https://img.shields.io/badge/python-3.8.5+-blue)](https://www.python.org/downloads/release/python-3613/)
[![torch](https://img.shields.io/badge/torch-1.8.1+-green)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-0.8.2+-green)](https://pytorch.org/)

# Generative deep features
In recent years more and more GANs are being trained to solve the problem of image generation, since they offer stunning visual quality. One of the bigger disadvantages of GANs is the need to use large datasets to train on, which aren't easily available for every purpose. SinGAN[1] was introduced as a model that combats this disadvantage, by training on a single image alone, by using a multi-scale GANs architecture.

In parallel to that, different papers published in the last couple of years have already established the connection between the deep features of classification networks and the semantic content of images, such that we can define the visual content of an image by the statistics of its deep features. 

The goal of this students project is to research the capability of generating a completely new image with the same visual content of a single given natural image, by using unsupervised learning of a deep neural network without the use of a GAN. Instead, we choose to use distribution loss functions over the deep features (The outputs of VGG19's feature maps) of the generated and original images. Using a similar pyramidal structure to SinGAN, we succeeded in creating realistic and variable images in different sizes and aspects, that maintain both the global structure and the visual content of the source image, by using a single image, without using an adversarial loss. We also apply the method on several additional applications, taken from the image manipulation tasks of the original paper.

## Table of Contents
* [Requirements](#requirements)
* [Usage Example](#usage-example)
* [Team](#team)
* [Examples](#examples)
* [Sources](#sources)
* 
## Requirements
The code was tested on the following libraries:
- imageio 2.9.0
- matplotlib 3.3.4
- numpy 1.19.5
- pytorch 1.8.1
- scikit-image 0.18.1
- scikit-learn 0.24.1
- scipy 1.6.1
- torch 1.7.1+cu110
- torchvision 0.8.2+cu110

## Usage Example
### Training a model (of a single image)

```bash
python main.py --image_path <input_image_path>
```

Sample outputs will automatically be generated at each scale and at the end of the training. use `--help` for more information on the parameters.

### Applications
You can find in the code folder multiple scripts for generation of applications, namely:
```
├── code
│   ├── edges.py
│   ├── animation.py
│   ├── scaled_sample.py
│   ├── harmonization.py
│   ├── paint_to_image.py
```

all will require 2 parameters `--image_path` for the original image the model was trained on, and `--train_net_dir` for the path to the trained model folder.
The output is located inside the trained model directory, given to the script, under a suiting name (e.g., `<train_net_dir>/Harmonization` for the `harmonization.py` script)
Each test has its own parameters, refer to the `--help` of each script. The relevant arguments always appear on top of the help page as `optional arguments`.

## Team
Hila Manor and Da-El Klang  
Supervised by Tamar Rott-Shaham

## Examples
All examples were generated using PDL as the distribution loss function
![image](https://user-images.githubusercontent.com/53814901/138109575-115c3cb3-7838-48f0-be4b-07786f3f5754.png)
birds parameters: min_size=21, a=25, a_color=0
mountains parameters: min_size=19, a=35, a_color=1
colosseum parameters: min_size=19, a=10, a_color=1

![image](https://user-images.githubusercontent.com/53814901/138109948-50118785-2144-49d6-b0f8-da7d3cfa6859.png)
trees parameters: min_size=19, a=35, a_color=0
cows parameters: min_size=19, a=10, a_color=1

![image](https://user-images.githubusercontent.com/53814901/138110179-889f4e5f-1328-412c-81c9-e6ea0b93e120.png)
starry night parameters: injection at 7th scale out of 11, min_size=21, a=25, a_color=3
tree parameters: injection at 9th scale oout of 12, min_size=19, a=35, a_color=1

![image](https://user-images.githubusercontent.com/53814901/138110363-ef89243f-3b91-4271-984a-3eb0abbe3547.png)
parameters: min_size=19, a=35, a_color=1

[Animation from a single image example](https://www.youtube.com/watch?v=8LO6iTGUI5c)

## Sources
- T. R. Shaham, et al. [SinGAN](https://tamarott.github.io/SinGAN.htm) -  [“SinGAN: Learning a Generative Model from a Single NaturalImage”](https://arxiv.org/pdf/1905.01164.pdf), IEEE International Conference on Computer vision (ICCV), pp. 4570-4580, 2019. [GitHub](https://github.com/tamarott/SinGAN)
- R. Mechrez, et al. [“The Contextual Loss for Image Transformation with Non-Aligned Data”](https://arxiv.org/pdf/1803.02077.pdf), European Conference on Computer Vision (ECCV), 2018, pp. 768–783.
- L. A. Gatys, et al. [“Image Style Transfer Using Convolutional Neural Networks”](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf), IEEE conference on computer vision and pattern recognition, 2016, pp. 2414–2423.
- M. Delbracio, et al. [“Projected Distribution Loss for Image Enhancement”](https://arxiv.org/pdf/2012.09289.pdf), arXiv preprint arXiv:2012.09289, 2020.
-	L. Biewald, Experiment Tracking with Weights and Biases, 2020. 
