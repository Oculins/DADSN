#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yuxian Jiang
@Contact: yuxianjiang@sjtu.edu.cn
@Article: Morphology Prior Enhanced Teeth Segmentation for High Resolution Oral Scans
@Journal: Journal of Biomedical and Health Informatics (2025)
"""

import os
import numpy as np
import json
from tqdm import tqdm
import random
import argparse
import trimesh
from iops import align_scan
import shutil


def preprocess(opt):
    dataset_name = opt.dataset_name
    read_root = opt.data_root
    data_write_root = os.path.join(opt.save_root, dataset_name)
    os.makedirs(data_write_root, exist_ok=True)

    files = []
    for name in tqdm(os.listdir(read_root)):
        if name.endswith('.obj'):
            case_name = name.split('.')[0]
            mesh_read_path = os.path.join(read_root, f"{case_name}.obj")
            label_read_path = os.path.join(read_root, f"{case_name}.json")
            npy_write_path = os.path.join(data_write_root, f"{case_name}.npy")
            label_write_path = os.path.join(data_write_root, f"{case_name}.json")

            mesh = trimesh.load(mesh_read_path, process=False)
            mesh = align_scan(mesh)
            vertices = np.array(mesh.vertices)
            normals = np.array(mesh.vertex_normals)
            features = np.concatenate([vertices, normals], axis=1)
            np.save(npy_write_path, features)

            shutil.copy(label_read_path, label_write_path)

            file = {
                'mesh_path': mesh_read_path,
                'npy_path': npy_write_path,
                'label_path': label_write_path,
            }
            files.append(file)

    print(f"{dataset_name}: {len(files)}")
    
    random.shuffle(files)
    num_files = len(files)
    num_train = int(0.75 * num_files)
    train_files = files[:num_train]
    val_files = files[num_train:]

    with open(os.path.join(opt.save_root, f"{dataset_name}.json"), "w") as f:
        json.dump(
            {
                "train_names": train_files,
                "val_names": val_files,
                "test_names": [],
            },
            f,
            indent=4,
        )
    print(f"train: {len(train_files)}, val: {len(val_files)}")



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # args.dataset_name
    args.add_argument('--data_root', type=str, help='data root')
    args.add_argument('--save_root', type=str, default='data', help='save root')
    args.add_argument('--dataset_name', type=str, default='Teeth3DS', help='dataset name')
    opt = args.parse_args()
    preprocess(opt)
        



