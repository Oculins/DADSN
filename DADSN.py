"""
@Author: Yuxian Jiang
@Contact: yuxianjiang@sjtu.edu.cn
@Article: Morphology Prior Enhanced Teeth Segmentation for High Resolution Oral Scans
@Journal: Journal of Biomedical and Health Informatics (2025)
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as dist



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, return_idx=False):
    # x: (B, dim, np)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x[:, :3], k=k)
    index = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base   # idx: (B, np, k)
    idx = idx.view(-1)  # idx: B*np*k
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]  # (B*np*k, dim)
    feature = feature.view(batch_size, num_points, k, num_dims) # (B, np, k, dim)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, np, k, dim)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    if return_idx:
        return feature, index
    else:
        return feature      # (batch_size, 2*num_dims, num_points, k)


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def select_indices_from_gaussian_distribution(mean, std, M, k):
    
    B, N = mean.shape

    x_values = torch.arange(0, M, dtype=torch.float32, device=mean.device) / M 
    dist_normal = dist.Normal(mean.unsqueeze(-1), std.unsqueeze(-1)) 
    probs = torch.exp(dist_normal.log_prob(x_values)) 
    probs = probs / probs.sum(dim=-1, keepdim=True)
    selected_indices = torch.multinomial(probs.view(B * N, M), k, replacement=False)
    selected_indices = selected_indices.view(B, N, k)
    selected_indices = selected_indices.sort(dim=-1)[0]
    selected_probs = torch.gather(probs, dim=2, index=selected_indices)

    return selected_indices, probs


class GaussianBlock(nn.Module):
    def __init__(self, in_channels):
        super(GaussianBlock, self).__init__()
        self.conv_mean = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel_size=1, bias=False),
            nn.InstanceNorm1d(in_channels*2),
            nn.Conv1d(in_channels*2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_std = nn.Sequential(
            nn.Conv1d(in_channels, in_channels*2, kernel_size=1, bias=False),
            nn.InstanceNorm1d(in_channels*2),
            nn.Conv1d(in_channels*2, 1, kernel_size=1, bias=False),
            nn.Softplus()
        )

    def forward(self, x):
        mean = self.conv_mean(x)
        std = self.conv_std(x) + 0.1
        return mean, std



class DADConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pos_dim, k, reception):
        super(DADConv, self).__init__()
        self.k = k
        self.reception = reception
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_channels*2, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels*2+pos_dim*2, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_channels*2, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))
        self.gaussianblock = GaussianBlock(hidden_channels)

        
    def forward(self, x_s, x, x0, widx):
        x_s = get_graph_feature(x_s, k=self.k, return_idx=False)
        x_s = self.conv_s(x_s)
        x_s = x_s.max(dim=-1, keepdim=False)[0]
        cur_idx = widx[:, :, :self.reception]

        mean, std = self.gaussianblock(x_s)
 
        random_indices, probs = select_indices_from_gaussian_distribution(mean.squeeze(1), std.squeeze(1), self.reception, self.k)
        selected_widx = cur_idx.gather(2, random_indices)
        random_probs = probs.gather(2, random_indices)
        random_probs = random_probs.unsqueeze(1)
        
        x = torch.cat((x, x0), dim=1)
        x_p = get_graph_feature(x, k=self.k, idx=selected_widx)
        x_p = self.conv_p(x_p)
        x_p = (x_p * random_probs).sum(dim=-1) / random_probs.sum(dim=-1)

        x = self.conv(torch.cat((x_s, x_p), dim=1))
        return x_s, x


class DADSN(nn.Module):
    def __init__(self, num_classes, k=32, emb_dims=1024, dropout=0.5, pos_dim=3):
        super(DADSN, self).__init__()
        
        self.k = k
        self.receptions = [k, k*4, k*16]

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2), 
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2))
        
        self.deformableconv1 = DADConv(64, 64, 64, pos_dim, self.k, self.receptions[1])
        self.deformableconv2 = DADConv(64, 64, 64, pos_dim, self.k, self.receptions[2])

        self.conv2 = nn.Sequential(
            nn.Conv1d(64*3, emb_dims, kernel_size=1, bias=False),
            nn.InstanceNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dims+64*3+pos_dim, 512, kernel_size=1, bias=False),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, bias=False),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv5 = nn.Conv1d(256+pos_dim, num_classes, kernel_size=1, bias=False)
    
    def forward(self, x):
        bs = x.size(0)
        npoint = x.size(2)
        device = x.device
        x0 = x[:, :3, :]
        widx = knn(x0, k=self.receptions[-1])

        x = get_graph_feature(x, k=self.k, idx=widx[:, :, :self.k])
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x_s, x2 = self.deformableconv1(x1, x1, x0, widx)
        x_s, x3 = self.deformableconv2(x_s, x2, x0, widx)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, npoint)
        x = torch.cat((x, x1, x2, x3, x0), dim=1)
        x = self.conv3(x)

        x = self.conv4(x)
        x = self.dp1(x)

        x = torch.cat((x, x0), dim=1)
        x = self.conv5(x)
        x = x.transpose(2, 1).contiguous()

        return x

