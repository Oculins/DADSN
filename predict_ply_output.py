
import argparse
import os
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
import json
from DADSN import DADSN
import trimesh
from util import get_random_index_patches, map_tooth_label
from iops import align_scan


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=16000, help='point number [default: 4096]') ###
    parser.add_argument('--num_classes', type=int, default=17, help='Number of classes') ###
    parser.add_argument('--ckp_path', type=str, default='best_model.pth', help='experiment root')  ###
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=32, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--expand_times', type=float, default=1)
    parser.add_argument('--work_dir', type=str, default='work/', help='experiment directory')
    return parser.parse_args()



def predict(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.work_dir
    input_dir = experiment_dir + '/input/'
    output_dir = experiment_dir + '/output/'

    '''LOG'''
    NUM_CLASSES = args.num_classes
    patch_point = args.npoint
    BATCH_SIZE = 1
    binarize = NUM_CLASSES == 2
    expand_times = args.expand_times

    file_list = os.listdir(input_dir)

    '''MODEL LOADING'''
    classifier = DADSN(num_classes=NUM_CLASSES, k=args.k, emb_dims=args.emb_dims).cuda()

    checkpoint = torch.load(args.ckp_path)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier = classifier.eval()

    color_dict = json.load(open('data/color_dict.json', 'r'))

    with torch.no_grad():

        for i, file_name in enumerate(file_list):
            case_name = file_name.split('.')[0]
            print(f"{i+1}/{len(file_list)}, Processing {case_name}")

            mesh_path = os.path.join(input_dir, file_name)
            
            mesh = trimesh.load(mesh_path)
            mesh = align_scan(mesh)
            vertices = np.asarray(mesh.vertices)
            normals = np.asarray(mesh.vertex_normals)

            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            mesh_o3d.compute_vertex_normals()

            num_all = len(vertices)

            patch_indices = get_random_index_patches(num_all, patch_point, expand=expand_times)
            # print(num_all, len(patch_indices))
            pred_prob = np.zeros((num_all, NUM_CLASSES), dtype=np.float32)
            pred_times = np.zeros(num_all, dtype=np.float32)

            for selected_point_idxs in patch_indices:

                ori_points = vertices[selected_point_idxs]
                ori_normals = normals[selected_point_idxs]

                normalized_points = (ori_points - ori_points.mean(axis=0)) / ori_points.std(axis=0)
                points = np.concatenate([normalized_points, ori_normals], axis=1)

                points = torch.Tensor(points)
                points = points.float().cuda()
                points = points.unsqueeze(0)
                points = points.transpose(2, 1)
                seg_pred = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                pred_val = pred_val[0]
                pred_prob[selected_point_idxs] += pred_val
                pred_times[selected_point_idxs] += 1


            pred_prob /= pred_times[:, None]
            output = np.argmax(pred_prob, axis=1)

            color_matrix = [color_dict[str(l)] for l in output]
            color_matrix = np.array(color_matrix)/255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_matrix)
            mesh_o3d.compute_vertex_normals()

            o3d.io.write_triangle_mesh(os.path.join(str(output_dir), f"{case_name}_seg.ply"), mesh_o3d, write_vertex_colors=True)
            # np.save(os.path.join(str(output_dir), f"{case_name}.ply"), output)


        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    predict(args)
