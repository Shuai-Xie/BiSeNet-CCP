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
from utils import get_label_info, one_hot_it
import random


def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass


def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass


class SUN(torch.utils.data.Dataset):
    def __init__(self, image_path, depth_path, label_path, csv_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.image_list = glob.glob(os.path.join(image_path, '*.jpg'))
        self.image_name = [x.split('/')[-1].split('.')[0] for x in self.image_list]
        self.depth_list = [os.path.join(depth_path, x + '.png') for x in self.image_name]
        self.label_list = [os.path.join(label_path, x + '.png') for x in self.image_name]
        self.fliplr = iaa.Fliplr(0.5)
        self.label_info = get_label_info(csv_path)
        # resize
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        self.resize_depth = transforms.Resize(scale, Image.NEAREST)
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and resize
        img = Image.open(self.image_list[index])
        img = self.resize_img(img)
        img = np.array(img)

        # load depth and resize
        depth = Image.open(self.depth_list[index])
        depth = self.resize_depth(depth)
        depth = np.array(depth)
        depth = depth[:, :, np.newaxis]  # add axis (480,640,1)

        # load label and resize
        label = Image.open(self.label_list[index])
        label = self.resize_label(label)
        label = np.array(label)

        # convert label to one-hot graph
        label = one_hot_it(label, self.label_info).astype(np.uint8)

        # augment image and label
        if self.mode == 'train':
            seq_det = self.fliplr.to_deterministic()
            img = seq_det.augment_image(img)
            depth = seq_det.augment_image(depth)
            label = seq_det.augment_image(label)

        # image -> to_tensor [3, H, W]
        img = Image.fromarray(img).convert('RGB')
        img = self.to_tensor(img).float()

        # depth -> to_tensor [1, H, W]
        depth = depth / 65535
        depth = self.to_tensor(depth).float()

        # image + depth = RGBD
        rgbd = torch.cat((img, depth), 0)

        # label -> [num_classes, H, W]
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)
        label = torch.from_numpy(label)

        return rgbd, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    data = SUN('/temp_disk/xs/sun/train/image', '/temp_disk/xs/sun/train/label_img', '/temp_disk/xs/sun/seg37_class_dict.csv', (480, 640))
    from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

    label_info = get_label_info('/temp_disk/xs/sun/seg37_class_dict.csv')
    for i, (img, label) in enumerate(data):
        print(img.shape)
        print(label.shape)
        print()
