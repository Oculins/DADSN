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
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def get_rotation_matrix(vector_source, vector_target):    # rotate B to A

    # Calculate the rotation axis by taking the cross product of the two vectors
    rotation_axis = np.cross(vector_source, vector_target)

    # Calculate the angle between the two vectors
    cos_angle = np.dot(vector_source, vector_target) / (np.linalg.norm(vector_source) * np.linalg.norm(vector_target))
    angle = np.arccos(cos_angle)

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) if np.linalg.norm(rotation_axis) != 0 else rotation_axis
    x, y, z = rotation_axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    rotation_matrix = np.array([
        [t*x**2 + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y**2 + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z**2 + c]
    ])

    return rotation_matrix

def parabola_3Dfitting(vertices, normals):
    init_normal = np.mean(np.asarray(normals), axis=0)
    init_normal /= np.linalg.norm(init_normal)
    init_rotation = get_rotation_matrix(np.array([0, 0, 1]), init_normal)
    init_rotation = R.from_matrix(init_rotation)
    init_angles = init_rotation.as_euler('xyz', degrees=True)
    init_offset = np.mean(np.asarray(vertices), axis=0)
    init_params = init_angles.tolist() + init_offset.tolist() + [1]

    def fitting_error(params, points):
        pitch, yaw, roll, x, y, z, a = params

        euler_angles = np.array([pitch, yaw, roll])
        rotation = R.from_euler('xyz', euler_angles, degrees=True)

        rot_z = rotation.apply(np.array([0, 0, 1]))
        A, B, C, D = rot_z[0], rot_z[1], rot_z[2], -np.dot(rot_z, np.array([x, y, z]))

        normal_length = np.sqrt(A ** 2 + B ** 2 + C ** 2)
        signed_distance_numerator = A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D
        signed_distances = signed_distance_numerator / normal_length
        projected_points = points - np.dot(signed_distances[:, None], np.array([[A, B, C]])) / normal_length

        trans_matrix = np.vstack([np.hstack([rotation.as_matrix(), np.array([[x], [y], [z]])]), np.array([[0, 0, 0, 1]])])
        trans_points = np.dot(np.linalg.inv(trans_matrix), np.vstack([projected_points.T, np.ones((1, points.shape[0]))]))
        trans_points = trans_points[:3, :].T

        distance_error = np.mean(signed_distances ** 2)
        parabola_error = np.mean((trans_points[:, 1] - a * trans_points[:, 0] ** 2) ** 2)

        center_distance_error = np.linalg.norm(np.mean(trans_points[:, :2], axis=0)) ** 2

        return distance_error * 10 + parabola_error + center_distance_error * 0.01

    optimized_params = minimize(
        fun=fitting_error,
        x0=init_params,
        args=(np.asarray(vertices)),
        method='L-BFGS-B',
    )

    pitch, yaw, roll, x, y, z, a = optimized_params.x
    euler_angles = np.array([pitch, yaw, roll])
    rotation = R.from_euler('xyz', euler_angles, degrees=True)
    rot_z = rotation.apply(np.array([0, 0, 1]))
    if np.dot(rot_z, init_normal) < 0:
        rot_z = - rot_z
    rot_y = rotation.apply(np.array([0, -1, 0])) if a >= 0 else rotation.apply(np.array([0, 1, 0]))
    center = np.array([x, y, z])

    return center, rot_z, rot_y

def align_scan(mesh, mask=None, sample_num=2000, target_orit=None, target_face=None):
    """
    Align the mesh to the standard orientation. center: [0, 0, 0], orit: [0, 0, -1], face: [1, 0, 0]
    :param mesh: Mesh
    :param mask: mask of vertices
    :param sample_num:
    :return: aligned mesh
    """
    vertices = mesh.vertices
    normals = np.asarray(mesh.vertex_normals)

    if mask is not None:
        mask = mask.astype(bool)
        # print(f"number of vertices before mask: {vertices.shape[0]}")
        vertices = vertices[mask]
        normals = normals[mask]
        # print(f"number of vertices after mask: {vertices.shape[0]}")

    if vertices.shape[0] > sample_num:
        indices = np.random.choice(vertices.shape[0], sample_num, replace=False)
        selected_vertices, selected_normals = vertices[indices], normals[indices]
        mesh_center, mesh_orit, mesh_face = parabola_3Dfitting(vertices[indices], normals[indices])
    else:
        selected_vertices, selected_normals = vertices, normals
        mesh_center, mesh_orit, mesh_face = parabola_3Dfitting(vertices, normals)

    np.set_printoptions(precision=3, suppress=True)
    # print(f"Mesh center: {mesh_center}, orientation: {mesh_orit}, face: {mesh_face}")
    center_new = np.array([0, 0, 0])

    def loss_function(rotation_vector, x1, y1, x2, y2):
        r = R.from_rotvec(rotation_vector).as_matrix()
        rot_x1 = np.dot(r, x1)
        rot_x2 = np.dot(r, x2)
        loss = np.sum((rot_x1 - y1) ** 2) + np.sum((rot_x2 - y2) ** 2)
        return loss
    
    target_orit = np.array([0, 0, -1]) if target_orit is None else np.asarray(target_orit)
    target_face = np.array([1, 0, 0]) if target_face is None else np.asarray(target_face)

    initial_guess = [0, 0, 0]
    bounds = [(-np.pi, np.pi) for _ in range(3)]
    result = minimize(fun=loss_function, x0=initial_guess,
                      args=(mesh_orit, target_orit, mesh_face, target_face),
                      method='L-BFGS-B', bounds=bounds)
    rotation_vector_optimal = result.x
    rotation_matrix = R.from_rotvec(rotation_vector_optimal).as_matrix()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = center_new - np.dot(rotation_matrix, mesh_center)
    mesh.apply_transform(transform_matrix)

    return mesh

