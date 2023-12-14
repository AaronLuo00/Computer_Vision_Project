
import os
import torch
import config
import argparse
import matplotlib.pyplot as plt

from utils import load_transformed_batch, load_rgb_batch, lab_to_rgb, load_generator

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='model type to train or test, specify one of [\'1_generator_base_l1_loss\', \'2_generator_base_fml_loss\', \'3_generator_base_l1_and_fml_loss\', \'4_generator_resnet_l1_loss\', \'5_generator_residual_unet_l1_loss\',  \'6_generator_base_l1_loss_pretrained\']',
                    type=str,
                    default='1_generator_base_l1_loss')
args = parser.parse_args()

if args.model in ['1_generator_base_l1_loss', '2_generator_base_fml_loss', '3_generator_base_l1_and_fml_loss']:
    config.GENERATOR_TYPE = 'UNet'
    
if args.model == '4_generator_resnet_l1_loss':
    config.GENERATOR_TYPE = 'ResNet'
    
if args.model == '5_generator_residual_unet_l1_loss':
    config.GENERATOR_TYPE = 'ResidualUNet'
    
if args.model == '6_generator_base_l1_loss_pretrained':
    config.GENERATOR_TYPE = 'PretrainedUNet'
    config.LOAD_PRETRAINED_GENERATOR = True
    config.PRETRAIN_GENERATOR = False

# %%

# Root directory for test-data
test_dir = os.path.join(os.getcwd(), 'test-images')
test_files = os.listdir(test_dir)

# Set the location of test-results directory and create the directory if it does not exists
res_dir = os.path.join(os.getcwd(), 'test-results')
os.makedirs(res_dir, exist_ok=True)

# Create generator object and load pretrained weights
generator = load_generator(config.GENERATOR_TYPE)
generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, '1_generator_base_l1_loss.pth'), map_location=config.DEVICE))

generator.eval()  # Since we are using only for testing
generator.requires_grad_(False)

# Load L and ab channels from the test-images
L, ab = load_transformed_batch(test_dir, test_files, config.VAL_TRANSFORMS)

if config.ENHANCE_COLORIZED_IMAGE:
    rgb_images = load_rgb_batch(config.TEST_DIR, test_files, config.UPSAMPLE_TRANSFORMS)
   
# L channel + Generator's ab channels make fake images
if config.ENHANCE_COLORIZED_IMAGE:
    fake_images = generator(L).permute(0, 2, 3, 1).detach().numpy()
else:
    fake_images = lab_to_rgb(L, generator(L))

# L channel + ground-truth ab channels make real images
real_images = lab_to_rgb(L, ab)

annotations = ["L Image", "Real Image", "Fake Image"]

for i in range(len(test_files)):
    
    fig = plt.figure(figsize=(50, 150))

    # Refactor subplot creation into a loop
    for j in range(1, 4):
        ax = plt.subplot(10, 5, j)

        # Use a conditional to decide which image to show
        if j == 1:
            img = L[i][0].cpu()
            ax.imshow(img, cmap='gray')
        elif j == 2:
            img = real_images[i]
            ax.imshow(img)
        elif j == 3:
            img = fake_images[i]
            ax.imshow(img)

        ax.axis("off")
        ax.set_title(annotations[j-1], fontsize=20)  # Add an annotation

    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig(os.path.join(res_dir, test_files[i]), bbox_inches='tight')
    plt.close()