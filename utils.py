import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import pandas as pd
import os


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=300, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


# return lr

def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {'null': [0, 0, 0]}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def one_hot_class_num(label):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for i in range(151):  # 0-150
        equality = np.equal(label, i)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.uint8)
    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
    # Returns
        Colour coded image for segmentation visualization
    """
    label_values = [label_values[key] for key in label_values]  # rgb
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


# from RedNet
def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        os._exit(0)
