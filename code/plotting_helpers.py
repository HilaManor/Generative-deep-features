import matplotlib.pyplot as plt
import numpy as np
import os

import image_processing
import image_helpers

def plot_losses(style_losses, rec_losses, ignore_first_iter=500, show=True):
    """
	The function creates ans shows a figure with 4 plots of the 2 given loss lists.
	
	:param style_losses: The distribution loss list, contains loss value from each epoch.
	:param rec_losses: The reconstruction loss list, contains loss value from each epoch.
	:param ignore_first_iter: The amount of skipped samples in the begining of the lists.
				The loss value decreases dramatically during the first iterations,
				so skipping them makes the graph more informative. Only applies 
				to the last 2 graphs.
	:param show: Boolean paramter, toggle between showing the graph during 
			runtime or not.
	
	:return: a generated figure object.
	"""
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
    # plt.plot(np.arange(len(style_losses))[ignore_first_iter:],
    #          (np.array(style_losses) + np.array(rec_losses))[ignore_first_iter:])
    plt.plot(np.arange(len(style_losses))[ignore_first_iter:],
             np.array(style_losses)[ignore_first_iter:], label='Distribution Loss')
    plt.title('Ignore start: Distribution Loss vs Iter')

    fig.add_subplot(2, 2, 4)
    # plt.plot(np.arange(len(style_losses))[ignore_first_iter:],
    #          np.array(style_losses)[ignore_first_iter:], label='Style Loss')
    plt.plot(np.arange(len(rec_losses))[ignore_first_iter:],
             np.array(rec_losses)[ignore_first_iter:], label='Reconstruction Loss', color='orange')
    plt.title('Ignore start: Reconstruction Loss vs Iter')

    if show:
        plt.show(block=False)
        plt.pause(0.01)
    return fig

def save_fig(fig, out_dir, name):
    """
	The function saves a given figure as png.
	
	:param fig: The given figure to save.
	:param out_dir: The desired directory for the output file.
	:param name: The desired name for the output file.
	
	"""
    path = os.path.join(out_dir, name + "_fig.png")
    fig.savefig(path)

def show_im(img_tensor, title=None):
    """
	The function shows a pytorch image, and saves a copy as np array format.
	
	:param img_tensor: The given pytorch image tensor.
	:param title: Optional. The title of the image, will be shown in the figure.
	
	:return: numpy version of the image.
	"""
    new_image = img_tensor.detach().to(device='cpu', copy=True)
    new_image = image_helpers.convert_image_np(new_image)
    plt.imshow(new_image)
    if title:
        plt.title(title)
    plt.show(block=False)
    plt.pause(0.01)
    return new_image

def save_im(img, out_dir, name, convert=False):
    """
	The function saves an image to the given dir. a pytorch image into np array format.
	
	:param img: The given image. np or pytorch image.
	:param out_dir: The desired directory for the output file.
	:param name: The desired name for the output file.
	:param convert: Boolean. if img is np.array, it should be set to False.
		If img is pytorch tensor, it should be set to True.
	
	
	:return: numpy version of the image.
	"""
    if convert:
        img = convert_im(img)

    path = os.path.join(out_dir, name + ".png")
    plt.imsave(path, img)

def convert_im(img):
    """
	The function convert a pytorch image into np array format.
	
	:param img: The given pytorch image tensor.
	
	:return: numpy version of the image.
	"""
    img = img.detach().to(device='cpu', copy=True)
    return image_helpers.convert_image_np(img)