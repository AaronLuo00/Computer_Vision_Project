# Enhancing Image Colorization: Residual UNet and Feature Matching in cGANs

This project focuses on advancing image colorization techniques using Residual UNet architecture and feature matching in conditional Generative Adversarial Networks (cGANs). 

## Dataset
The dataset used is MS-COCO, which can be downloaded from the [MS-COCO Dataset website](https://cocodataset.org/#download). For convenience, a selection of test images has been provided in the `test-images` directory.

## Model Dictionary
A sample of the model dictionary is available for download. You can access it [here](https://drive.google.com/drive/folders/12LUdwi967VBjObwKcw-FIBQZyyunjAk0?usp=sharing).

## Modules
The project is structured into independent modules, each module can be ran independently, facilitating ease of use:
- `train.py`: Main module for training the model.
- `test.py`: Module for testing the model.
- `eval.py`: Module for evaluating the model and calculating metrics.
- `config.py`: Manages configuration parameters for the entire project. Modify this file to adjust hyperparameters and project settings.
- `fm_loss.py`: Defines the feature matching loss function
- `evaluation_metrics.py`: Contains metrics for evaluating the performance of the colorization models
- `generator.py`: The generator part of the GAN, including UNet, ResNet, ResUNet.
- `discriminator.py`: The discriminator part of the GAN.
- `utils.py`: A utility module containing helper functions

## Configuration
To train or test with different hyperparameters, modify the `config.py` file accordingly. This allows for easy experimentation and tuning.

## Acknowledgments
Part of the code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [Towards Data Science](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8).

