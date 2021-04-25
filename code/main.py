from config import get_arguments
import torch
import random
import os
import functions
import training

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()

    opt.layers_weights = [1, 0.75]
    opt.chosen_layers = ['conv1_1', 'conv2_1']
    opt.is_cuda = opt.is_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda:0" if opt.is_cuda else "cpu")
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, consider removing --not_cuda")

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    basename = os.path.basename(opt.image_path)
    basename = basename[:basename.rfind('.')]
    out_dir = os.path.join(opt.output, basename)
    os.makedirs(out_dir, exist_ok=True)

    real_img = functions.read_image(opt.image_path, opt.nc, opt.is_cuda)
    real_img = functions.resize(real_img,
                                min(opt.max_size / max([real_img.shape[2], real_img.shape[3]]), 1),
                                opt.nc, opt.is_cuda)

    training.train(real_img, out_dir, opt)

