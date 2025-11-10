
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from util import map_tooth_label

def augment_mesh(vertices, normals, rotation_range=np.pi, translation_range=10):
    theta = np.random.uniform(-rotation_range, rotation_range, size=3)
    theta[:2] = 0
    rotation = R.from_euler('xyz', theta)
    rotation_matrix = rotation.as_matrix()

    vertices = np.dot(vertices, rotation_matrix.T)
    normals = np.dot(normals, rotation_matrix.T)

    translation = np.random.uniform(-translation_range, translation_range, size=3)
    translation[:2] = 0
    vertices += translation
    return vertices, normals


class TeethDataset(Dataset):
    def __init__(self, file_paths: list[str], num_classes, num_points, test=False):
        self.len = len(file_paths)
        self.file_paths = file_paths
        self.num_points = num_points
        self.num_classes = num_classes
        self.test = test

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        npy_path = file_path['npy_path']
        label_path = file_path['label_path']

        features = np.load(npy_path)  # Nx6
        vertices, normals = features[:, :3], features[:, 3:]

        gt = json.load(open(label_path, 'r'))
        label = np.array(gt['vertex_labels'])
        
        label = map_tooth_label(label)
        assert label.max() < self.num_classes

        N_points = vertices.shape[0]

        assert N_points == len(label)

        if N_points >= self.num_points:
            selected_point_idxs = np.random.choice(N_points, self.num_points, replace=False)
        else:
            selected_point_idxs = np.random.choice(N_points, self.num_points, replace=True)

        selected_points = vertices[selected_point_idxs, :]
        selected_normals = normals[selected_point_idxs, :]
        current_labels = label[selected_point_idxs]
        selected_features = features[selected_point_idxs, :]

        selected_points_normalized = (selected_points - selected_points.mean(axis=0)) / selected_points.std(axis=0)
        selected_features = np.concatenate((selected_points_normalized, selected_normals), axis=1)

        if self.test:
            return selected_features, current_labels, selected_points
        else:
            return selected_features, current_labels


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, power=2):
        super().__init__()
        self.smooth = smooth
        self.power = power

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, d1, ..., dn), probability
        y: (batch_size, d1, ..., dn), binary label
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1) ** self.power
        y = y.view(batch_size, -1).float()

        intersection = (x * y).sum(1)
        unionset = x.sum(1) + y.sum(1)
        dice = (2 * intersection + self.smooth) / (unionset + self.smooth)
        dice = dice.sum() / batch_size
        loss = 1 - dice
        return loss        


# input shape [N, d1, ..., dn]
# output shape [N, C, d1, ..., dn]
class OneHot:
    def __init__(self, num_classes=-1) -> None:
        self.num_classes = num_classes

    def __call__(self, class_index_label: torch.Tensor) -> torch.Tensor:
        shape_len = len(class_index_label.size())
        class_one_hot_label: torch.Tensor = F.one_hot(
            class_index_label, self.num_classes
        )
        class_one_hot_label = class_one_hot_label.permute(
            0, shape_len, *range(1, shape_len)
        )
        return class_one_hot_label


class MultiDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, power=2, num_classes=17, weight=None):
        super().__init__()
        self.binary_dice = DiceLoss(smooth=smooth, power=power)
        self.softmax = torch.nn.Softmax(dim=1)
        self.get_one_hot = OneHot(num_classes=num_classes)
        if weight is None:
            self.weight = np.ones(num_classes) / num_classes
        else:
            self.weight = weight / np.sum(weight)

    def forward(self, result: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        result: (batch_size, num_classes, d1, ..., dn)
        label: (batch_size, d1, ..., dn), value in [0, num_classes - 1]
        """
        result = self.softmax(result)
        label = self.get_one_hot(label)

        num_classes = label.size(1)
        total_dice_loss = 0
        for i in range(num_classes):
            total_dice_loss += self.binary_dice(result[:, i], label[:, i]) * self.weight[i]

        return total_dice_loss



