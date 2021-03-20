# Artwork Generation Using Deep Convolutional GAN, Conditional GAN and Creative Adversarial Network

This repository contains 3 GAN models to generate realistic artwork paintings. The models are implemented using PyTorch.

## WikiArt Dataset

The WikiArt dataset can be downloaded from [this link](https://drive.google.com/file/d/1uX3rC7a_aRtsQAz8UTCGOiTvO-3b0yXY/view?usp=sharing) (resized to 64x64)

The original WikiArt dataset is contained in [this repo](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset).

## References

The models in this repository are the implementations of the following papers:

* [Deep Convolutional GAN (DCGAN) Paper](https://arxiv.org/pdf/1511.06434.pdf): A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convo-lutional generative adversarial networks,”arXiv preprint arXiv:1511.06434, 2015.

* [Conditional GAN (CGAN) Paper](https://arxiv.org/pdf/1411.1784.pdf): M. Mirza and S. Osindero, “Conditional generative adversarial nets,”arXiv preprintarXiv:1411.1784, 2014.

* [Creative GAN (CAN) Paper](https://arxiv.org/pdf/1706.07068.pdf): A. Elgammal, B. Liu, M. Elhoseiny, and M. Mazzone, “Can:  Creative adversarial networks,generating  “art”  by  learning  about  styles  and  deviating  from  style  norms,”arXiv preprintarXiv:1706.07068, 2017.

[This PyTorch tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) was extremely helpful to develop our models.

## Models

* DCGAN

The DCGAN architecture is our baseline for creating realistic artwork paintings. Below are some examples generated by our network.

* CGAN

The CGAN architecture enables style-specific artwork generation by feeding the discriminator and the generator with artistic style labels. Below are some examples that belong to several artistic style classes.

![](https://github.com/otepencelik/GAN-Artwork-Generation/blob/master/cgan_results.png)

* CAN

The CAN architecture aims to generate style-ambiguous (or style-agnostic, 'creative') artwork pieces. The discriminator has access to artistic style labels. During training, the generator is punished if the discriminator correctly classifies the artistic style of a fake artwork. The generator is therefore pushed to generate more creative artwork that can't be classified into any of the artistic styles. Below are some creative fake artwork pieces generated by our network.

![](https://github.com/otepencelik/GAN-Artwork-Generation/blob/master/CAN_examples.png)

## Usage

Three very straight-forward notebooks are available for each of the models. Run each cell of the notebook to train the corresponding architecture and visualize the results.

* DCGAN - [Baseline_DCGAN.ipynb](https://github.com/otepencelik/GAN-Artwork-Generation/blob/master/Baseline_DCGAN.ipynb)
* CGAN - [cGAN.ipynb](https://github.com/otepencelik/GAN-Artwork-Generation/blob/master/cGAN.ipynb)
* CAN - [CAN.ipynb](https://github.com/otepencelik/GAN-Artwork-Generation/blob/master/CAN.ipynb)


