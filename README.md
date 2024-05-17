# ImageColoring
Coloring black and white images using deep learning and PyTorch.

The code is found in Colab in the link provided:
https://colab.research.google.com/drive/133oRIf8J-Qe3QejWyXuhKz4DxPkb9dJn?usp=sharing

You can train the model yourself from scratch, or download the pretrained weights and use it to colorize your black and white images.



## Final model's output

![image](https://github.com/ElironLubaton/ImageColoring/assets/125808481/6e33047e-619a-4456-846e-b1856904ff41)

Top row: Greyscale version of the images from test set.
Middle row: Colorized outputs by our model.
Bottom row: Real images from the test set.

___
![image](https://github.com/ElironLubaton/ImageColoring/assets/125808481/76841ba6-b9bf-4e5a-a9b2-aab33e412291)
Top row: Greyscale version of the images from test set.
Middle row: Colorized outputs by our model.
Bottom row: Real images from the test set.

___


## Dataset
We have used the Flowers102 dataset:   https://www.robots.ox.ac.uk/~vgg/data/flowers/102/  
The dataset consists of 8,819 colored pictures of flowers and provides us with a diverse range of flower types in different colors, shapes, picture angels and backgrounds.



## Pre Processing
The pre processing was done in several steps:
#### 1 - Resizing the images to 128X128.
We've managed to train the model properly also with a resize of 256X256, but the training process was much longer.

#### 2 - Converting the images from RGB color space to L*a*b color space.
This is the most important step which improved our results significantly. While working in RGB color space, it is needed to convert the three channels into one channel (grey channel) and then predict three channels again, hoping the image would be colored correctly. Thus, assuming we have 256 choices for each channel of the RGB, we have 256³ different combinations.

While using L*a*b color space, each letter represents:
L - Represents lightness from black to white on a scale of 0 to 100.
a* - On a scale of -110 to 110, negative corresponds with green, positive corresponds with red.
b* - On a scale of -110 to 110, negative corresponds with blue, positive corresponds with yellow.

When using L*a*b color space, we can give the L channel to the model (which is the greyscale image) and we want it to predict the other two channels, a* and b*. Thus, we have 220 choices for two channels, meaning we have $220^2$ different combinations, which is far smaller than $256^3$, providing us with better results and a great improvement in the model's training time.

#### 3 - Data Augmentation
Each time we accesed a batch of images, we randomly flipped the images horizontally with a 50% chance.

#### 4 - Spliting the data
We've splited the data into training, validation and test, with the proportions of 70%, 15% and 15%.



## Architecture
![WGAN ARCHITECTURE](https://github.com/ElironLubaton/ImageColoring/assets/125808481/005bc5bb-5712-43ff-829d-9ee02f715a03)

We have used WGAN - Wasserstein-generative adversarial network (Martin Arjovsky, 2017)  with gradient penalty (Ishaan Gulrajani, 2017) and L1 loss.

#### Generator
Our generator is based upon the U-net architecture (Olaf Ronneberger, 2015).
It consists of encoder-decoder structure:
The encoder part of the U-net down-samples the input image, extracting features at multiple scales each time.
This part is composed of 4 blocks, where each block is composed of 2 continues blocks of convolution, batch normalization and ReLU, and between each block a maxpooling is used.

The decoder part of the U-net up-samples the feature representation back to the original image size, while incorporating skip connections to retain the small details.
This part is composed of 4 blocks, where each block is composed of a transposed convolution, and 2 continues blocks of convolution, batch normalization and ReLU.

![ENCODER DECODER](https://github.com/ElironLubaton/ImageColoring/assets/125808481/5dd78750-b257-47de-af83-4a6f6417de23)

#### Critic (Discriminator)
Our critic is a 'patch discriminator' (Phillip Isola, 2018).
It consists of 5 blocks, where each block is composed of convolution, instance normalization and leaky ReLU, where only the first block lacks the normalization.
The parameters of the critic were initialized with normal distribution with a mean of 0, and standard deviation of 0.02, as proposed by (Phillip Isola, 2018)

![CRITIC](https://github.com/ElironLubaton/ImageColoring/assets/125808481/2401ba9b-bceb-48ae-89db-ed82731543a3)


## Training The Model

#### Pre-Trainig the Generator
We chose to pre-train our generator only in order to give it a head start, because in early runs the critic learned much faster than the generator, and the generator could not fool the critic. 
The pre-training was done using L1 loss between the colorized images and the original images.

![Pretraining](https://github.com/ElironLubaton/ImageColoring/assets/125808481/d5695240-a15d-4577-ba3c-aab2561a6f87)

#### Training the Model
The algorithm we've used is WGAN with Gradient penalty as proposed by (Ishaan Gulrajani, 2017), with some modifications.
In the training process, the generator and the critic were trained alternately, such that the generator was trained twice on each batch, and the critic was trained only once on each batch, for a total of 110 epochs.

The generator was trained by using 2 terms that were backpropagated:
	L1 loss between the colorized images and the original images.
	The critic's loss on the colorized images, which is the mean of the critic's output, where its input is colorized images.

The critic was trained by using 2 terms that were backpropagated:
	Wasserstein distance – the difference between the critic's loss on the colorized images and the real images.
	A gradient penalty term multiplied by the hyper-parameter λ.

#### Evaluation
The generator performance was evaluated by using the peak signal-to-noise ratio.


## Results

#### Results from validation set while training




## Bibliography

Ishaan Gulrajani, F. A. (2017). Improved Training of Wasserstein GANs. Retrieved from Arxiv: https://arxiv.org/pdf/1704.00028.pdf

Martin Arjovsky, S. C. (2017). Wasserstein GAN. Retrieved from Arxiv: https://arxiv.org/pdf/1701.07875.pdf

Olaf Ronneberger, P. F. (2015). U-Net: Convolutional Networks for Biomedical. Retrieved from Arxiv: https://arxiv.org/pdf/1505.04597v1.pdf

Phillip Isola, J.-Y. Z. (2018). Image-to-Image Translation with Conditional Adversarial Networks. Retrieved from Arxiv: https://arxiv.org/pdf/1611.07004.pdf


Moein Shariatnia, (2018), Colorizing black & white images with U-Net and conditional GAN. Retrieved from Towards Data Science: https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8

