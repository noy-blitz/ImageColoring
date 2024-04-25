import torch
from torch import optim
from torch.utils.data import DataLoader

import glob
import os
from fastai.data.external import untar_data, URLs, tarfile
import warnings

from Model import *
from preprocessing import *
from train import *
from utilities import *


def plotting():
    # Generator Train
    plot_loss(gen_L1_losses, 'L1 Loss',
              gen_WGAN_losses, 'Critic loss on colorized images',
              gen_losses, 'Generator Loss: L1 loss + Critic loss',
              mode = "Training - Generator")

    # Generator Validation
    plot_loss(val_gen_L1_losses, 'L1 Loss',
              val_gen_WGAN_losses, 'Critic loss on colorized images',
              val_gen_losses, 'Generator Loss: L1 loss + Critic loss',
              mode = "Validation - Generator")

    # Critic Wasserstein Loss
    plot_loss(critic_WGAN_losses, 'Training',
              val_critic_WGAN_losses, 'Validation',
              mode = 'Wasserstein loss')

    # Critic Wasserstein Loss with GP
    plot_loss_gp(critic_losses, 'WGAN Critic Loss with GP')

    # PSNR values
    plot_psnr(psnr_values, mode='Evaulation Set')


warnings.filterwarnings("ignore")


# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test=True - test the model
# test=False - train the model
test = True
# Choosing the path based the platform you are using
path = '/'#'/content/'      # path when using Google Colab
# path = '/kaggle/working/' # path when using Kaggle


seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# Downloading the dataset
untar_data(URLs.FLOWERS, (path + "data"), base=(path + 'data'))
tarfile.open(path + "data/oxford-102-flowers.tgz", 'r:gz').extractall(path + 'data')
images = glob.glob(path + 'data/oxford-102-flowers/jpg' + "/*.jpg")


# Creating directories in order to store the images
pretrain_dir = path + 'generated_images/pretrain' # pre-training image directory
os.makedirs(pretrain_dir, exist_ok=True)
train_dir = path + 'generated_images/training' # training image directory
os.makedirs(train_dir, exist_ok=True)
val_dir = path + 'generated_images/validation' # validation image directory
os.makedirs(val_dir, exist_ok=True)
test_dir = path + 'generated_images/test' # test image directory
os.makedirs(test_dir, exist_ok=True)


# Deciding the proportions of how to split the data
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

train_split_size = int(len(images) * TRAIN_SPLIT)
val_split_size = int(len(images) * VAL_SPLIT)
test_split_size = int(len(images) * TEST_SPLIT)

# Spliting the images randomly
rand_idxs = np.random.permutation(len(images))

train_idxs = list(rand_idxs[:train_split_size])
train_paths = [images[i] for i in train_idxs]

val_idxs = list(rand_idxs[train_split_size: train_split_size + val_split_size])
val_paths = [images[i] for i in val_idxs]

test_idxs = list(rand_idxs[train_split_size + val_split_size:])
test_paths = [images[i] for i in test_idxs]

# Creating the datasets and Dataloaders
train_set = Datasets(train_paths)
trainloader_color = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)

validation_set = Datasets(val_paths)
validationloader_color = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                    num_workers=2)

test_set = Datasets(test_paths)
testloader_color = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)


# Creating the model
# Training Lists
# Generator
gen_L1_losses = []       # Saves the L1 losses of the generator
gen_WGAN_losses = []      # Saves the GAN losses of the generator
gen_losses = []          # Saves both L1 loss and GAN loss of the generator

# Critic
critic_real_losses = []  # Saves the critic loss of the REAL images
critic_fake_losses = []  # Saves the critic loss of the FAKE images
critic_WGAN_losses = []  # Saves the critic loss for BOTH real and fake images
critic_losses = []       # Saves the critic loss for BOTH real and fake images and GP


# Validation lists
# Generator
val_gen_L1_losses = []       # Saves the L1 losses of the generator
val_gen_WGAN_losses = []      # Saves the GAN losses of the generator
val_gen_losses = []          # Saves both L1 loss and GAN loss of the generator

# Critic
val_critic_real_losses = []  # Saves the critic loss of the REAL images
val_critic_fake_losses = []  # Saves the critic loss of the FAKE images
val_critic_WGAN_losses = []  # Saves the critic loss for BOTH real and fake images

psnr_values = []


# Initializing Networks and Optimizers
generator = Generator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.9))

critic = Critic().to(device)
optimizer_C = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.0, 0.9))

# Defining loss function
loss_function = torch.nn.L1Loss()


if __name__ == '__main__':


    #Training the model:
    if test:  # Testing the model
        print("before test_model")
        test_model(testloader_color)


    else:  # Training the model
        # Set to False if you don't want to use pre-training
        pretrain = True

        # If you want to further train parameters, set to True
        use_weights = False

        if pretrain:
            print("Pre-training the Generator")
            pretrain_generator(epochs=5)
            print("___________________________________\n\n\n\n")

            # Loading the generator's parameters from the pre-training
            gen_weights = torch.load(path + 'pretrained_generator.pth', map_location=device)
            generator.load_state_dict(gen_weights['model_weights'])
            optimizer_G.load_state_dict(gen_weights['optimizer_weights'])

        if use_weights:  # Further training existing parameters
            # Generator
            gen_weights = torch.load(path + 'trained_generator.pth', map_location=device)
            generator.load_state_dict(gen_weights['model_weights'])
            optimizer_G.load_state_dict(gen_weights['optimizer_weights'])
            # Critic
            critic_weights = torch.load(path + 'trained_critic.pth', map_location=device)
            critic.load_state_dict(critic_weights['model_weights'])
            optimizer_C.load_state_dict(critic_weights['optimizer_weights'])

        # Training Loop
        print("Training the Model\n")
        train_wgan_gp(epochs=110, generator_iterations=2, lambda_gp=10)

        plotting()