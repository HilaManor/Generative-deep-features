import matplotlib.pyplot as plt
import numpy as np
import os
import re

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

def __generate_out_name(opt):
    base_name = f"{opt.loss_func}_e{opt.epochs}_{opt.nzx}px_lr{opt.lr}" \
                 f"a{opt.alpha}_up-{opt.chosen_layers[-1]}_W{opt.layers_weights}"
    return base_name


def gen_unique_out_dir_path(base_out, baseim_name, opt):
    possible_name = __generate_out_name(opt)
    possible_base = os.path.join(base_out, baseim_name)
    possible_path = os.path.join(possible_base, possible_name)

    if os.path.exists(possible_path):
        # rename with "name_name(num)"
        dirs = [f for f in os.listdir(possible_base) if os.path.isdir(os.path.join(possible_base, f))]

        ptrn = possible_name.replace('[', '\[').replace(']', '\]')
        matches = re.findall(ptrn+r'(\((\d+)\))?', '\n'.join(dirs))
        int_matches = [int(j) for i,j in matches if j]
        if int_matches:
            possible_name += f'({max(int_matches)+1})'
        else:
            possible_name += '(1)'

        possible_path = os.path.join(possible_base, possible_name)
    return possible_path
