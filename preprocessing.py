import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 128  # Controls the resizing of the image
BATCH_SIZE = 64 # Controls the batch size

class Datasets(Dataset):
    """
    An instance of this class is created for each dataset (train, val, test).
    Each time an image is fetched, it is converted from RGB to L*a*b.
    """
    def __init__(self, data):
        self.transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC),
            transforms.RandomHorizontalFlip()
        ])
        self.data = data


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Loading an image, transform it and turn it to numpy array
        rgb_image = Image.open(self.data[idx])
        rgb_image = rgb_image.convert("RGB")
        rgb_image = self.transforms(rgb_image)
        rgb_image = np.array(rgb_image)

        # Converting the image from RGB to L*a*b
        lab_img = rgb2lab(rgb_image)
        lab_img = lab_img.astype("float32")
        lab_img = transforms.ToTensor()(lab_img)

        # Normalizing L*a*b values
        L, ab = self.normalize_lab(lab_img)

        return {'L': L, 'ab': ab}


    def normalize_lab(self, lab_img):
        # Normalizing L and ab values
        L = (lab_img[0:1, ...] / 50) - 1
        ab = (lab_img[1:3, ...] / 110)
        return L, ab


