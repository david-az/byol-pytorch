import torch
from torch import nn
from torchvision import transforms as T
import torchvision.transforms.functional as F 

class NormalizeMeanVar(nn.Module):
    def __call__(self, img):
        return (img - img.mean([1, 2, 3], True)) / img.std([1, 2, 3], keepdim=True)

class NormalizeMinMax(nn.Module):
    def __call__(self, img):
        return (img - img.min()) / (img.max() - img.min())

class BitDepthConversion(nn.Module):
    """Convert the image bit depth.
    Args:
        imgbit (int): Desired bit depth for the image.
    """

    def __init__(self, imgbit=8):
        self.imgbit = imgbit

    def __call__(self, sample):
        """Call function to convert the image bit depth.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img' key is modified and 'image_bit_depth'
                key is added (dict with original bit depth and converted).
        """
        img = sample

        # get image bit depth
        input_img_bit = torch.log2(torch.max(img) + 1)

        # compute the bit difference
        bit_diff = input_img_bit - self.imgbit

        if bit_diff > 0:
            # need to decrease the bit-depth
            divide_by = 2**bit_diff
            if self.imgbit == 8:
                img = ((img / divide_by)).astype('uint8')
            elif self.imgbit > 8:
                img = ((img / divide_by)).astype('uint16')
        else:
            # need to increase the bit-depth
            multiply_by = 2**(-bit_diff)
            if self.imgbit == 8 and img.max() < 256:
                img = img.astype('uint8')
            else:
                img = ((img * multiply_by)).astype('uint16')

        return img


class RandomHorizontalOrVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            if torch.rand(1) < 0.5:
                img = F.hflip(img)
            else:
                img = F.vflip(img)
        return img

def pipeline_nojitter(image_size):
    transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-20, 20), expand=False),
            NormalizeMeanVar()
        ])
    return transform

def pipeline_nojitter_lighter(image_size):
    transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.9, 1.)),
            RandomHorizontalOrVerticalFlip(),
            T.RandomRotation(degrees=(-10, 10)),
            NormalizeMeanVar()
        ])
    return transform

def pipeline_jitter(image_size):
    transform = T.Compose([
            NormalizeMinMax(),
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.)),
            T.ColorJitter(.2, .2, .0, .0),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-20, 20), expand=False),
            NormalizeMeanVar()
        ])
    return transform
