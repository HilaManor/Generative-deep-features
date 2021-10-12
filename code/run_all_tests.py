import os


## animation

#birds
for folder in [os.listdir('code\Output\birds')]:
    if not os.path.isfolder(folder):
        continue
    for alpha in [0.4,0.6,0.8]:
        for fps in [10, 25]:
            os.sytem(f"animation.py --image_path ..\images\2\birds.png --trained_net_dir \"{folder}\" --animation_fps {fps} --animation_alpha {alpha} --animation_initial_beta_sweep 0.05 --animation_final_beta_sweep 0.99")

#lightining
for folder in [os.listdir('code\Output\lightning1')]:
    if not os.path.isfolder(folder):
        continue
    for alpha in [0.4,0.6,0.8]:
        for fps in [10, 25]:
            os.sytem(f"animation.py --image_path ..\images\2\lightning1.png --trained_net_dir \"{folder}\" --animation_fps {fps} --animation_alpha {alpha} --animation_initial_beta_sweep 0.05 --animation_final_beta_sweep 0.99")


## paint to image

#cows
for folder in [os.listdir('code\Output\cows')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [1,2,3]:
        for qunat in [""," --quantization_flag"]:
            os.system(f"--image_path ..\images\2\cows.png --trained_net_dir \"{folder}\" --ref_path ..\images\paint\cows.png --paint_start_scale {scale}{quant}")

#mountains
for folder in [os.listdir('code\Output\mountains')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [1,2,3]:
        for qunat in [""," --quantization_flag"]:
            os.system(f"--image_path ..\images\2\mountains.jpg --trained_net_dir \"{folder}\" --ref_path ..\images\paint\mountains_paint_reflect.png --paint_start_scale {scale}{quant}")
#trees
for folder in [os.listdir('code\Output\trees3')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [1,2,3]:
        for qunat in [""," --quantization_flag"]:
            os.system(f"--image_path ..\images\2\trees3.png --trained_net_dir \"{folder}\" --ref_path ..\images\paint\trees1.png --paint_start_scale {scale}{quant}")

## harmonization

#seascape
for folder in [os.listdir('code\Output\seascape')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [3,4,5,6,7,8,9]:
            os.system(f"--image_path ..\images\2\seascape.png --trained_net_dir \"{folder}\" --ref_path ..\images\harmonization\tree.jpg --mask_path ..\images\harmonization\tree_mask.jpg --harmonization_start_scale {scale}")

#starry night
for folder in [os.listdir('code\Output\starry_night_full')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [3,4,5,6,7,8,9]:
            os.system(f"--image_path ..\images\1\starry_night_full.jpg --trained_net_dir \"{folder}\" --ref_path ..\images\harmonization\starry_night_naive.png --mask_path ..\images\harmonization\starry_night_naive_mask.png --harmonization_start_scale {scale}")

## Editing

# stone
for folder in [os.listdir('code\Output\stone')]:
    if not os.path.isfolder(folder):
        continue
    for scale in [1,2,3]:
            os.system(f"--image_path ..\images\2\stone.png --trained_net_dir \"{folder}\" --ref_path ..\images\harmonization\stone_edit.png --mask_path ..\images\harmonization\stone_edit_mask.png --editing --harmonization_start_scale {scale}")
