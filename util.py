import os
import glob
import copy
import random
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def map_tooth_label(label: np.ndarray) -> np.ndarray:
    """
    fdi to continuous label
    11~18 -> 1~8
    21~28 -> 9~16
    31~38 -> 1~8
    41~48 -> 9~16
    """
    label[label > 30] -= 20
    label[label > 10] -= 10
    label[label > 10] -= 2
    return label

def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.0)
        except AttributeError:
            pass


def bn_momentum_adjust(m, momentum):
    if isinstance(m, nn.BatchNorm2d) or \
            isinstance(m, nn.BatchNorm1d):
        m.momentum = momentum


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    target[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()



def get_random_index_patches(N, patch_size, expand=1):
    if N <= patch_size:
        additional_points = patch_size - N
        additional_indices = np.random.choice(N, additional_points, replace=True)
        selected = np.concatenate([np.arange(N), additional_indices])
        return [selected]

    indices = np.arange(N)
    np.random.shuffle(indices)

    patches = []
    while len(indices) >= patch_size:
        patch = indices[:patch_size]
        patches.append(patch)
        indices = indices[patch_size:]  # 更新剩余点

    if len(indices) > 0:
        additional_points = patch_size - len(indices)
        remaining_indices = np.setdiff1d(np.arange(N), indices, assume_unique=True)
        additional_indices = np.random.choice(remaining_indices, additional_points, replace=False)
        patch = np.concatenate([indices, additional_indices])
        patches.append(patch)

    assert len(patches) == N // patch_size + (N % patch_size > 0)

    expand_num = int((expand - 1) * len(patches)) if expand > 1 else 0
    for i in range(expand_num):
        patch = np.random.choice(N, patch_size, replace=False)
        patches.append(patch)
    
    return patches

