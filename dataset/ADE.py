import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from utils import get_label_info, one_hot_class_num
import random


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


class ADE(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, scale, mode='train'):
        """
        :param image_path: /home/disk1/xs/ADEChallengeData2016/images/training
        :param label_path: /home/disk1/xs/ADEChallengeData2016/annotations/training
        :param csv_path: /home/disk1/xs/ADEChallengeData2016/ade150.csv
        :param scale: (args.crop_height, args.crop_width)
        :param mode: train, test
        """
        super().__init__()
        self.mode = mode
        self.ori_image_list = glob.glob(os.path.join(image_path, '*.jpg'))
        if mode == 'train':
            # random choose 2000 images to train
            rand_idxs = list(np.random.permutation(20210))
            # rand_idxs = rand_idxs[:10000]
        else:
            # random choose 100 images to val
            rand_idxs = list(np.random.permutation(2000))
            # rand_idxs = rand_idxs[:2000]

        self.image_list = [self.ori_image_list[idx] for idx in rand_idxs]
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        self.label_list = [os.path.join(label_path, x + '.png') for x in self.image_name]
        self.fliplr = iaa.Fliplr(0.5)
        # resize
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_list[index])
        img = self.resize_img(img)
        img = np.array(img)

        # load label
        label = Image.open(self.label_list[index])
        label = self.resize_label(label)
        label = np.array(label)

        # convert label to one-hot graph
        label = label[:, :, np.newaxis]
        label = one_hot_class_num(label)

        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            label = seq_det.augment_image(label)

        # image -> [C, H, W]
        img = Image.fromarray(img).convert('RGB')
        img = self.to_tensor(img).float()

        # label -> [num_classes, H, W]
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)
        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data = ADE(image_path='/home/disk1/xs/ADEChallengeData2016/images/training',
               label_path='/home/disk1/xs/ADEChallengeData2016/annotations/training',
               scale=(480, 640))

    for i, (img, label) in enumerate(data):
        print(img.shape)
        print(label.shape)
        break
