import os
import os.path as osp
import pickle
import numpy as np
import math
import random
from scipy.spatial import cKDTree
from typing import Any, Dict, List, Tuple, Union
import yaml
from ast import literal_eval
import hashlib
import json
import torch
from typing import Optional, Callable
import torch.distributed as dist
from collections import OrderedDict
import open3d as o3d
import logging
import coloredlogs
from scipy.spatial.transform import Rotation
from torch.autograd import Function
# from cpp_openpoints.pointnet2_batch import pointnet2_cuda
from multimethod import multimethod



'''calculate model size'''
def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total



'''dirs'''
def ensure_dir(path):
    if not osp.exists(path):
        os.makedirs(path)


def remove_state_dict_head(state_dict):
    from collections import OrderedDict
    model_state_dict = OrderedDict([(key.split('.', 1)[1], value) for key, value in state_dict.items()])
    return model_state_dict

def make_log_dirs(log_cfg):
    log_path=log_cfg.logdirs
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    event_dirs=os.path.join(log_path,'event')
    snapshot_dirs=os.path.join(log_path,'snapshot')
    debug_dirs=os.path.join(log_path,'debug')
    log_dirs=os.path.join(log_path,'logs')
    ensure_dir(event_dirs)
    ensure_dir(snapshot_dirs)
    ensure_dir(debug_dirs)
    ensure_dir(log_dirs)
    log_cfg.update(dict(event_dirs=event_dirs))
    log_cfg.update(dict(snapshot_dirs=snapshot_dirs))
    log_cfg.update(dict(debug_dirs=debug_dirs))
    log_cfg.update(dict(log_dirs=log_dirs))


'''pickle'''
def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


'''output format'''
def get_print_format(value):
    if isinstance(value, int):
        return 'd'
    if isinstance(value, str):
        return 's'
    if value == 0:
        return '.3f'
    if value < 1e-6:
        return '.3e'
    if value < 1e-3:
        return '.6f'
    return '.3f'


def get_format_strings(kv_pairs):
    r"""Get format string for a list of key-value pairs."""
    log_strings = []
    for key, value in kv_pairs:
        fmt = get_print_format(value)
        format_string = '{}: {:' + fmt + '}'
        log_strings.append(format_string.format(key, value))
    return log_strings


'''model utils'''
def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


'''data processing utils'''
def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals

def make_open3d_point_cloud(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def voxel_downsample(points, voxel_size, normals=None):
    pcd = make_open3d_point_cloud(points, normals=normals)
    pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)
    if normals is not None:
        normals = np.asarray(pcd.normals)
        return points, normals
    else:
        return points


def regularize_normals(points, normals, positive=True):
    r"""Regularize the normals towards the positive/negative direction to the origin point.

    positive: the origin point is on positive direction of the normals.
    negative: the origin point is on negative direction of the normals.
    """
    dot_products = -(points * normals).sum(axis=1, keepdims=True)
    direction = dot_products > 0
    if positive:
        normals = normals * direction - normals * (1 - direction)
    else:
        normals = normals * (1 - direction) - normals * direction
    return normals


def normalize_points(points):
    np.seterr(divide='ignore', invalid='ignore')    # if it not exits, may warning: RuntimeWarning: invalid value encountered in divide
    r"""Normalize point cloud to a unit sphere at origin."""
    points = points - points.mean(axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))
    return points

# def translate_pointcloud(pointcloud):
#     xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
#     xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
#
#     translated_pointcloud = np.add(np.multiply(
#         pointcloud, xyz1), xyz2).astype('float32')
#     return translated_pointcloud
#
#
# def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
#     N, C = pointcloud.shape
#     pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
#     return pointcloud
#
#
# def rotate_pointcloud(pointcloud):
#     theta = np.pi * 2 * np.random.uniform()
#     rotation_matrix = np.array(
#         [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#     pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
#         rotation_matrix)  # random rotation (x,z)
#     return pointcloud

# class FurthestPointSampling(Function):
#     @staticmethod
#     def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
#         """
#         Uses iterative furthest point sampling to select a set of npoint features that have the largest
#         minimum distance
#         :param ctx:
#         :param xyz: (B, N, 3) where N > npoint
#         :param npoint: int, number of features in the sampled set
#         :return:
#              output: (B, npoint) tensor containing the set (idx)
#         """
#         assert xyz.is_contiguous()
#
#         B, N, _ = xyz.size()
#         # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
#         # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
#         output = torch.cuda.IntTensor(B, npoint)
#         temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
#
#         pointnet2_cuda.furthest_point_sampling_wrapper(
#             B, N, npoint, xyz, temp, output)
#         return output
#
#     @staticmethod
#     def backward(xyz, a=None):
#         return None, None
#
#
# furthest_point_sample = FurthestPointSampling.apply

# def fps(data, number):
#     '''
#         data B N C
#         number int
#     '''
#     fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
#     fps_data = torch.gather(
#         data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
#     return fps_data


def fps_manual(xyz, npoint, normal=None):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    xyz = torch.gather(
        xyz, 1, centroids.unsqueeze(-1).long().expand(-1, -1, xyz.shape[-1]))
    if normal is not None:
        normal = torch.gather(
            normal, 1, centroids.unsqueeze(-1).long().expand(-1, -1, normal.shape[-1]))
        return xyz, normal
    return xyz



def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances

def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


@multimethod
def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points

@multimethod
def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points


def apply_rotation(points: torch.Tensor, rotation: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rotate points and normals (optional) along the origin.

    Given a point cloud P(3, N), normals V(3, N) and a rotation matrix R, the output point cloud Q = RP, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), rotation is (3, 3), the output points are (*, 3).
       In this case, the rotation is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 3, 3), the output points are (B, N, 3).
       In this case, the rotation is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        rotation (Tensor): (3, 3) or (B, 3, 3)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if rotation.ndim == 2:
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2))
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif rotation.ndim == 3 and points.ndim == 3:
        points = torch.matmul(points, rotation.transpose(-1, -2))
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and rotation{}.'.format(tuple(points.shape), tuple(rotation.shape))
        )
    if normals is not None:
        return points, normals
    else:
        return points


def sample_points(points, num_samples, normals=None):
    r"""Sample the first K points."""
    points = points[:num_samples]
    if normals is not None:
        normals = normals[:num_samples]
        return points, normals
    else:
        return points


def random_sample_points(points, num_samples, normals=None):
    r"""Randomly sample points."""
    num_points = points.shape[0]
    sel_indices = np.random.permutation(num_points)
    if num_points > num_samples:
        sel_indices = sel_indices[:num_samples]
    elif num_points < num_samples:
        num_iterations = num_samples // num_points
        num_paddings = num_samples % num_points
        all_sel_indices = [sel_indices for _ in range(num_iterations)]
        if num_paddings > 0:
            all_sel_indices.append(sel_indices[:num_paddings])
        sel_indices = np.concatenate(all_sel_indices, axis=0)
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points


def random_scale_shift_points(points, low=2.0 / 3.0, high=3.0 / 2.0, shift=0.2, normals=None):
    r"""Randomly scale and shift point cloud."""
    scale = np.random.uniform(low=low, high=high, size=(1, 3))
    bias = np.random.uniform(low=-shift, high=shift, size=(1, 3))
    points = points * scale + bias
    if normals is not None:
        normals = normals * scale
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return points, normals
    else:
        return points


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.
    Args:
        rotation (array): (3, 3)
        translation (array): (3,)
    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform

def inverse_transform(transform: np.ndarray) -> np.ndarray:
    r"""Inverse rigid transform.

    Args:
        transform (array): (4, 4)

    Return:
        inv_transform (array): (4, 4)
    """
    rotation, translation = get_rotation_translation_from_transform(transform)  # (3, 3), (3,)
    inv_rotation = rotation.T  # (3, 3)
    inv_translation = -np.matmul(inv_rotation, translation)  # (3,)
    inv_transform = get_transform_from_rotation_translation(inv_rotation, inv_translation)  # (4, 4)
    return inv_transform


def random_rotate_points_along_up_axis(points, normals=None):
    r"""Randomly rotate point cloud along z-axis."""
    theta = np.random.rand() * 2.0 * math.pi
    # fmt: off
    rotation_t = np.array([
        [math.cos(theta), math.sin(theta), 0],
        [-math.sin(theta), math.cos(theta), 0],
        [0, 0, 1],
    ])
    # fmt: on
    points = np.matmul(points, rotation_t)
    if normals is not None:
        normals = np.matmul(normals, rotation_t)
        return points, normals
    else:
        return points


def random_rescale_points(points, low=0.8, high=1.2):
    r"""Randomly rescale point cloud."""
    scale = random.uniform(low, high)
    points = points * scale
    return points


def random_jitter_points(points, scale, noise_magnitude=0.05):
    r"""Randomly jitter point cloud."""
    noises = np.clip(np.random.normal(scale=scale, size=points.shape), a_min=-noise_magnitude, a_max=noise_magnitude)
    points = points + noises
    return points


def random_shuffle_points(points, normals=None):
    r"""Randomly permute point cloud."""
    indices = np.random.permutation(points.shape[0])
    points = points[indices]
    if normals is not None:
        normals = normals[indices]
        return points, normals
    else:
        return points


def random_dropout_points(points, max_p):
    r"""Randomly dropout point cloud proposed in PointNet++."""
    num_points = points.shape[0]
    p = np.random.rand(num_points) * max_p
    masks = np.random.rand(num_points) < p
    points[masks] = points[0]
    return points


def random_jitter_features(features, mu=0, sigma=0.01):
    r"""Randomly jitter features in the original implementation of FCGF."""
    if random.random() < 0.95:
        features = features + np.random.normal(mu, sigma, features.shape).astype(np.float32)
    return features


def random_sample_plane():
    r"""Random sample a plane passing the origin and return its normal."""
    phi = np.random.uniform(0.0, 2 * np.pi)  # longitude
    theta = np.random.uniform(0.0, np.pi)  # latitude

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    normal = np.asarray([x, y, z])

    return normal


def random_crop_point_cloud_with_plane(points, p_normal=None, keep_ratio=0.7, normals=None):
    r"""Random crop a point cloud with a plane and keep num_samples points."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if p_normal is None:
        p_normal = random_sample_plane()  # (3,)
    distances = np.dot(points, p_normal)
    sel_indices = np.argsort(-distances)[:num_samples]  # select the largest K points
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points

def random_sample_viewpoint(limit=500):
    r"""Randomly sample observing point from 8 directions."""
    return np.random.rand(3) + np.array([limit, limit, limit]) * np.random.choice([1.0, -1.0], size=3)

def random_crop_point_cloud_with_point(points, viewpoint=None, keep_ratio=0.7, normals=None):
    r"""Random crop point cloud from the observing point."""
    num_samples = int(np.floor(points.shape[0] * keep_ratio + 0.5))
    if viewpoint is None:
        viewpoint = random_sample_viewpoint()
    distances = np.linalg.norm(viewpoint - points, axis=1)
    sel_indices = np.argsort(distances)[:num_samples]
    points = points[sel_indices]
    if normals is not None:
        normals = normals[sel_indices]
        return points, normals
    else:
        return points



'''training utils'''
class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.normal_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step):
        # last_step starts from -1, which means last_steps=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        else:
            if next_step > self.total_steps:
                return self.eta_min
            next_step -= self.warmup_steps
            return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * next_step / self.normal_steps))


def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler


def get_log_string(result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, timer=None, loader_id=None):
    log_strings = []
    if epoch is not None:
        epoch_string = f'Epoch: {epoch}'
        if max_epoch is not None:
            epoch_string += f'/{max_epoch}'
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f'iter: {iteration}'
        if max_iteration is not None:
            iter_string += f'/{max_iteration}'
        if epoch is None:
            iter_string = iter_string.capitalize()
        log_strings.append(iter_string)
    if 'metadata' in result_dict:
        log_strings += result_dict['metadata']
    for key, value in result_dict.items():
        if key != 'metadata':
            format_string = '{}: {:' + get_print_format(value) + '}'
            log_strings.append(format_string.format(key, value))
    if lr is not None:
        log_strings.append('lr: {:.3e}'.format(lr))
    if timer is not None:
        log_strings.append(timer.tostring())
    if loader_id is not None:
        log_strings.append('loader_id: {}'.format(loader_id))
    message = ', '.join(log_strings)
    return message



def write_event(writer, phase, event_dict, index, local_rank=0):
    r"""Write TensorBoard event."""
    if local_rank != 0:
        return
    for key, value in event_dict.items():
        writer.add_scalar(f'{phase}/{key}', value, index)

def save_register_snapshots(model, snapshot_dir, epoch,iteration, logger, optimizer, scheduler=None, local_rank=0, distributed=False, metric=None):
    if local_rank != 0:
        return

    model_state_dict = model.state_dict()
    # Remove '.module' prefix in DistributedDataParallel mode.
    if distributed:
        model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])

    if metric is None:
        filename=f'epoch-{epoch}_iter-{iteration}.pth.tar'
    else:
        assert isinstance(metric, dict), 'Metric TypeError'
        precision, error = round(metric['precision'], 4), round(metric['error'], 4)
        filename = f'epoch-{epoch}_iter-{iteration}_acc-{str(precision)}_err-{str(error)}.pth.tar'

    # save model
    filename = osp.join(snapshot_dir, filename)

    state_dict = {
        'epoch': epoch,
        'iteration': iteration,
        'model': model_state_dict,
    }
    torch.save(state_dict, filename)
    logger.info('Model saved to "{}"'.format(filename))

    # save snapshot
    snapshot_filename = osp.join(snapshot_dir, 'snapshot.pth.tar')
    state_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()
    torch.save(state_dict, snapshot_filename)
    logger.info('Snapshot saved to "{}"'.format(snapshot_filename))
#
# def save_snapshots(model, snapshot_dir, epoch, iteration, logger, optimizer, scheduler=None, local_rank=0, distributed=False, acc_t=None):
#     if local_rank != 0:
#         return
#
#     model_state_dict = model.state_dict()
#     # Remove '.module' prefix in DistributedDataParallel mode.
#     if distributed:
#         model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])
#
#     if acc_t==None:
#         filename=f'epoch-{epoch}_iter-{iteration}.pth.tar'
#     else:
#         acc_t=round(acc_t, 4)
#         filename=f'epoch-{epoch}_iter-{iteration}_acc-{str(acc_t)}.pth.tar'
#
#     # save model
#     filename = osp.join(snapshot_dir, filename)
#
#     state_dict = {
#         'epoch': epoch,
#         'iteration': iteration,
#         'model': model_state_dict,
#     }
#     torch.save(state_dict, filename)
#     logger.info('Model saved to "{}"'.format(filename))
#
#     # save snapshot
#     snapshot_filename = osp.join(snapshot_dir, 'snapshot.pth.tar')
#     state_dict['optimizer'] = optimizer.state_dict()
#     if scheduler is not None:
#         state_dict['scheduler'] = scheduler.state_dict()
#     torch.save(state_dict, snapshot_filename)
#     logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

# 'cuda tensor utils'
def to_cuda(x, rank):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item, rank) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item, rank) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value, rank) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.to(rank)
    return x

def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x

def all_reduce_tensor(tensor, world_size=1):
    r"""Average reduce a tensor across all workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor

def all_reduce_tensors(x, world_size=1):
    r"""Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x

def release_tensors(result_dict,distributed=False,world_size=1):
    r"""All reduce and release tensors."""
    if distributed:
        result_dict = all_reduce_tensors(result_dict, world_size)
    result_dict = release_cuda(result_dict)
    return result_dict

'loss'
def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae

def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae

def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


'''dict_config utils'''
class EasyConfig(dict):
    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    # def load(self, fpath: str, *, recursive: bool = False) -> None:
    def load(self, fpath: str,) -> None:
        """load cfg from yaml

        Args:
            fpath (str): path to the yaml file
            recursive (bool, optional): recursily load its parent defaul yaml files. Defaults to False.
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        fpaths = [fpath]
        # if recursive:
        #     extension = os.path.splitext(fpath)[1]
        #     while os.path.dirname(fpath) != fpath:
        #         fpath = os.path.dirname(fpath)
        #         fpaths.append(os.path.join(fpath, 'default' + extension))
        for fpath in reversed(fpaths):
            if os.path.exists(fpath):
                with open(fpath) as f:
                    self.update(yaml.safe_load(f))

    def load_model(self, cfg_backbone_dir):
        if not os.path.exists(cfg_backbone_dir):
            raise FileNotFoundError(cfg_backbone_dir)
        # if recursive:
        #     extension = os.path.splitext(fpath)[1]
        #     while os.path.dirname(fpath) != fpath:
        #         fpath = os.path.dirname(fpath)
        #         fpaths.append(os.path.join(fpath, 'default' + extension))
        with open(cfg_backbone_dir) as f:
            self.update(yaml.safe_load(f))

    def reload(self, fpath: str, *, recursive: bool = False) -> None:
        self.clear()
        self.load(fpath, recursive=recursive)

    # mutimethod makes python supports function overloading
    @multimethod
    def update(self, other: Dict) -> None:
        for key, value in other.items():
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], EasyConfig):
                    self[key] = EasyConfig()
                # recursively update
                self[key].update(value)
            else:
                self[key] = value


    @multimethod
    def update(self, opts: Union[List, Tuple]) -> None:
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith('--'):
                opt = opt[2:]
            if '=' in opt:
                key, value = opt.split('=', 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split('.')
            try:
                value = literal_eval(value)
            except:
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, EasyConfig())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        configs = dict()
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                value = value.dict()
            configs[key] = value
        return configs

    def hash(self) -> str:
        buffer = json.dumps(self.dict(), sort_keys=True)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, EasyConfig):
                seperator = '\n'
            else:
                seperator = ' '
            text = key + ':' + seperator + str(value)
            lines = text.split('\n')
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (' ' * 2) + line
            texts.extend(lines)
        return '\n'.join(texts)


'''Logger'''
def create_logger(log_file=None):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level=logging.DEBUG)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'

    stream_handler = logging.StreamHandler()
    colored_formatter = coloredlogs.ColoredFormatter(format_str)
    stream_handler.setFormatter(colored_formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

class Logger:
    # def __init__(self, log_file=None, local_rank=-1):
    def __init__(self, log_file=None):
        # if local_rank == 0 or local_rank == -1:
        #     self.logger = create_logger(log_file=log_file)
        # else:
        #     self.logger = None
        try:
            self.logger = create_logger(log_file=log_file)
        except:
            print('[ERROR]  -----------------------------Init logger---------------------------')
            exit(0)

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message):
        if self.logger is not None:
            self.logger.critical(message)



'''get config from x.yaml'''
def get_config_rgst(args,signal):
    cfg = EasyConfig()
    cfg.load(args.cfg)
    bkb_path=cfg.model.backbone
    cfg.model.backbone=EasyConfig()
    cfg.model.backbone.load_model(bkb_path)  # overwrite the model backbone
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    # cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    # cfg.sync_bn = cfg.world_size > 1

    cfg.dirs = EasyConfig()
    cfg.dirs.utils_working_dir = osp.dirname(osp.realpath(__file__))
    cfg.dirs.root_dir = osp.dirname(cfg.dirs.utils_working_dir)
    # cfg.dirs.data_root = osp.join(cfg.dirs.root_dir, 'data')
    if signal==2:
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_localRegister2')
    elif signal==3:
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_localRegister3_small')
    elif signal==4:
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_localRegister3_hardest')
    elif signal==5:
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_cls_modelnet40')
        cfg.dirs.best_val_cls_dir = osp.join(cfg.dirs.output_dir, 'best_val_cls')
        ensure_dir(cfg.dirs.best_val_cls_dir)
    elif signal==6:
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_test_0')




    cfg.dirs.snapshot_dir = osp.join(cfg.dirs.output_dir, 'snapshots')
    cfg.dirs.log_dir = osp.join(cfg.dirs.output_dir, 'logs')
    # cfg.dirs.event_dir = osp.join(cfg.dirs.output_dir, 'events')
    cfg.dirs.debug_dir = osp.join(cfg.dirs.output_dir, 'debugs')
    # cfg.dirs.feature_dir = osp.join(cfg.dirs.output_dir, 'features')
    # cfg.dirs.registration_dir = osp.join(cfg.dirs.output_dir, 'registration')

    ensure_dir(cfg.dirs.output_dir)
    ensure_dir(cfg.dirs.snapshot_dir)
    ensure_dir(cfg.dirs.log_dir)
    ensure_dir(cfg.dirs.debug_dir)
    # ensure_dir(cfg.dirs.feature_dir)
    # ensure_dir(cfg.dirs.registration_dir)

    return cfg

def get_config_task(args,signal):
    cfg = EasyConfig()
    cfg.load(args.cfg)
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    cfg.dirs = EasyConfig()
    cfg.dirs.utils_working_dir = osp.dirname(osp.realpath(__file__))
    cfg.dirs.root_dir = osp.dirname(cfg.dirs.utils_working_dir)
    # cfg.dirs.data_root = osp.join(cfg.dirs.root_dir, 'data')
    if signal==5:       # modelnet_cls
        cfg.dirs.output_dir = osp.join(cfg.dirs.root_dir, 'output_cls_modelnet40')
        cfg.dirs.best_val_cls_dir = osp.join(cfg.dirs.output_dir, 'best_val_cls')
        ensure_dir(cfg.dirs.best_val_cls_dir)
    elif signal==6:     # seg or other cls
        pass

    cfg.dirs.snapshot_dir = osp.join(cfg.dirs.output_dir, 'snapshots')
    cfg.dirs.log_dir = osp.join(cfg.dirs.output_dir, 'logs')
    # cfg.dirs.event_dir = osp.join(cfg.dirs.output_dir, 'events')
    cfg.dirs.debug_dir = osp.join(cfg.dirs.output_dir, 'debugs')
    # cfg.dirs.feature_dir = osp.join(cfg.dirs.output_dir, 'features')
    # cfg.dirs.registration_dir = osp.join(cfg.dirs.output_dir, 'registration')

    ensure_dir(cfg.dirs.output_dir)
    ensure_dir(cfg.dirs.snapshot_dir)
    ensure_dir(cfg.dirs.log_dir)
    ensure_dir(cfg.dirs.debug_dir)
    # ensure_dir(cfg.dirs.feature_dir)
    # ensure_dir(cfg.dirs.registration_dir)

    return cfg

