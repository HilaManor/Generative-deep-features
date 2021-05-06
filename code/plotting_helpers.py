import matplotlib.pyplot as plt
import numpy as np
import os

import image_processing

def plot_losses(style_losses, rec_losses, ignore_first_iter=500):
    fig = plt.figure(figsize=(16, 10))
    fig.add_subplot(2, 2, 1)
    plt.plot(np.array(style_losses) + np.array(rec_losses))
    plt.title('Total loss vs Iter')

    fig.add_subplot(2, 2, 2)
    plt.plot(style_losses, label='Style Loss')
    plt.plot(rec_losses, label='Reconstruction Loss')
    plt.title('Losses vs Iter')
    plt.legend()

    fig.add_subplot(2, 2, 3)
    plt.plot(np.arange(len(style_losses))[ignore_first_iter:],
             (np.array(style_losses) + np.array(rec_losses))[ignore_first_iter:])
    plt.title('Ingore start: Total loss vs Iter')

    fig.add_subplot(2, 2, 4)
    plt.plot(np.arange(len(style_losses))[ignore_first_iter:],
             np.array(style_losses)[ignore_first_iter:], label='Style Loss')
    plt.plot(np.arange(len(rec_losses))[ignore_first_iter:],
             np.array(rec_losses)[ignore_first_iter:], label='Reconstruction Loss')
    plt.title('Ingore start: Losses vs Iter')

    plt.show(block=False)
    plt.pause(0.01)
    return fig

def save_fig(fig, out_dir, name):
    path = os.path.join(out_dir, name + "_fig.png")
    fig.savefig(path)

def show_im(img_tensor, title=None):
    new_image = img_tensor.detach().to(device='cpu', copy=True)
    new_image = image_processing.convert_image_np(new_image)
    plt.imshow(new_image)
    if title:
        plt.title(title)
    plt.show(block=False)
    plt.pause(0.01)
    return new_image

def save_im(img, out_dir, name, convert=False):
    if convert:
        img = img.detach().to(device='cpu', copy=True)
        img = image_processing.convert_image_np(img)

    path = os.path.join(out_dir, name + ".png")
    plt.imsave(path, img)
