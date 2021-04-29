import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(losses, ignore_first_iter=1000, only_last_iter=5000):
    fig = plt.figure(figsize=(16, 10))
    fig.add_subplot(2, 2, 1)
    plt.plot(np.log10(losses))
    plt.title('log10(Losses) vs Iter')
    fig.add_subplot(2, 2, 2)
    plt.plot(np.arange(len(losses))[ignore_first_iter:], np.log10(losses[ignore_first_iter:]))
    plt.title('Zoomed log10(Losses) vs Iter')
    fig.add_subplot(2, 2, 3)
    plt.plot(np.arange(len(losses))[-only_last_iter:], losses[-only_last_iter:])
    plt.title('Zoomed Losses vs Iter')
    # fig.add_subplot(2, 2, 4)
    # plt.plot(np.arange(len(losses))[ignore_first_iter:], np.array(losses[ignore_first_iter:]) - np.array(losses[501:-1]))
    # plt.title('Zoomed Derivative vs Iter')
    return fig

def save_fig(fig, out_dir, opt):
    base_name = f"{opt.loss_func}_e{opt.epochs}_{opt.nzx}px_lr{opt.lr}" \
                 f"_up-{opt.chosen_layers[-1]}_W{opt.layers_weights}"
    path = os.path.join(out_dir, base_name + "_fig.png")
    fig.savefig(path)
