"""Data loader for ModelNet40
"""
import argparse, os, torch, h5py, torchvision
import os.path as osp
from typing import List, Optional
import numpy as np
from torch.utils.data import Dataset

from . import modelnet_transforms as Transforms
from src.utils.utils import load_pickle, estimate_normals, dump_pickle, normalize_points, random_sample_transform, \
    inverse_transform, apply_transform, random_crop_point_cloud_with_plane, random_sample_viewpoint, \
    random_crop_point_cloud_with_point, random_sample_points, random_jitter_points, random_shuffle_points, fps_manual, \
    compute_overlap

import sys

# sys.path.append(os.path.dirname(os.getcwd()))


def get_trainval_datasets(args: argparse.Namespace):
    # train_categories, val_categories = None, None
    # if args.train_categoryfile:
    #     train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
    #     train_categories.sort()
    # if args.val_categoryfile:
    #     val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
    #     val_categories.sort()

    train_transforms, val_transforms = get_transforms(args.setting.noise_type,
                                                      args.setting.rotation_magnitude,
                                                      args.setting.translation_magnitude,
                                                      args.setting.num_points,
                                                      args.setting.partial)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    # train_data = ModelNetHdf(args, args.root, subset='train', categories=train_categories,
    #                          transform=train_transforms)
    # val_data = ModelNetHdf(args, args.root, subset='test', categories=val_categories,
    #                        transform=val_transforms)
    args.setting.update(dict(split='train'))
    train_dataset = ModelNet40(train_transforms, args)

    args.setting.update(dict(split='val'))
    val_dataset = ModelNet40(val_transforms, args)

    return train_dataset, val_dataset


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),        # generate src, ref and corresp from  points
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class ModelNet40(Dataset):
    # fmt: off
    ALL_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
        'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
        'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
        'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_CATEGORIES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', 'chair', 'curtain', 'desk', 'door', 'dresser',
        'glass_box', 'guitar', 'keyboard', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
        'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'wardrobe', 'xbox'
    ]
    ASYMMETRIC_INDICES = [
        0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36,
        38, 39
    ]
    SYMMETRIC_CATEGORIES = ['bottle', 'bowl', 'cone', 'cup', 'flower_pot', 'lamp', 'tent', 'vase']
    SYMMETRIC_CATEGORIES_ID = [5, 6, 9, 10, 15, 19, 34, 37]

    # fmt: on

    def __init__(self, transform=None, args=None):
        super(ModelNet40, self).__init__()
        # self.args=args
        assert args.setting.split in ['train', 'trainval', 'val', 'test']

        self.data_root = args.modelnet40.data_root
        self.num_points = args.setting.num_points
        self.use_normal = args.setting.use_normal
        self.presample = args.setting.presample  # if you first run it, please set it 'True' to generate dataset.pkl
        self.asymmetric = args.modelnet40.asymmetric
        self.class_indices = self.get_class_indices(args.modelnet40.class_indices, args.modelnet40.asymmetric)
        self.split = args.setting.split
        self.transform = transform

        if self.split in ['train', 'trainval']:
            self.split = 'train'
        else:
            self.split = 'test'
        if self.asymmetric:
            pkl_name = os.path.join(self.data_root, f'modelnet40_reg_{self.split}.pkl')
        else:
            pkl_name = os.path.join(self.data_root, f'modelnet40_reg_symm_{self.split}.pkl')
        if self.presample and not os.path.exists(pkl_name):
            # store datas
            with open(os.path.join(self.data_root, 'modelnet40_ply_hdf5_2048', f'{self.split}_files.txt')) as f:
                lines = f.readlines()
            all_points = []
            all_normals = []
            all_labels = []
            for line in lines:
                filename = line.strip()
                # h5file = h5py.File(f'modelnet40_ply_hdf5_2048/{filename}', 'r')
                h5file = h5py.File(filename, 'r')
                # for key in h5file.keys():     # keys: data, faceId, label, normal--error
                #     print(key)
                all_points.append(h5file['data'][:])
                all_normals.append(h5file['normal'][:])
                all_labels.append(h5file['label'][:].flatten().astype(np.int32))
            points = np.concatenate(all_points, axis=0).astype(np.float32)
            # print(f'calculating modelnet40_reg {self.split} normals, please waiting...')
            # for point in points:
            #     normal = estimate_normals(point)
            #     all_normals.append(normal)  # raw data: error normal
            # print(f'-------------------estimate modelnet40_reg {self.split} normals end--------------------------')
            # normals = np.array(all_normals)
            normals = np.concatenate(all_normals, axis=0).astype(np.float32)
            labels = np.concatenate(all_labels, axis=0).astype(np.int64)
            if self.split == 'train':       # asymmetric
                all_points.clear()
                all_normals.clear()
                all_labels.clear()
                for i, label in enumerate(labels):
                    if label in self.class_indices:
                        all_points.append(points[i])
                        all_normals.append(normals[i])
                        all_labels.append(label)
                points = np.array(all_points)  # train--(8284,2048,3)      test--(2148,2048,3)
                normals = np.array(all_normals)  # train--(8284,2048,3)      test--(2148,2048,3)
                labels = np.array(all_labels)  # train--(8284,)        test--(2148,)

            # for label in labels:
            #     if label in self.SYMMETRIC_CATEGORIES_ID:
            #         print(label)

            self.data_dict = {'points': points, 'normals': normals, 'labels': labels}
            # save datas as pickle
            dump_pickle(self.data_dict, pkl_name)
            print(
                f"{osp.basename(pkl_name)} saved and load successfully. datas size is {len(self.data_dict['labels'])}")
        else:
            self.data_dict = load_pickle(pkl_name)
            print(f"{osp.basename(pkl_name)} load successfully. datas size is {len(self.data_dict['labels'])}")

    def __getitem__(self, item):

        sample = {
            'points': np.concatenate([self.data_dict['points'][:], self.data_dict['normals'][:]], axis=-1)[item, :, :] if self.use_normal else self.data_dict['points'][:][item, :, :],
            'label': self.data_dict['labels'][item], 'idx': np.array(item, dtype=np.int32)}

        # Apply perturbation
        if self.transform:
            sample = self.transform(sample)

        corr_xyz = np.concatenate([
            sample['points_src'][sample['correspondences'][0], :3],
            sample['points_ref'][sample['correspondences'][1], :3]], axis=1)

        # Transform to my format
        sample_out = {
            'src_xyz': torch.from_numpy(sample['points_src'][:, :3]),
            'tgt_xyz': torch.from_numpy(sample['points_ref'][:, :3]),
            'tgt_raw': torch.from_numpy(sample['points_raw'][:, :3]),
            'src_overlap': torch.from_numpy(sample['src_overlap']),
            'tgt_overlap': torch.from_numpy(sample['ref_overlap']),
            'correspondences': torch.from_numpy(sample['correspondences']),
            'pose': torch.from_numpy(sample['transform_gt']),
            'idx': torch.from_numpy(sample['idx']),
            'corr_xyz': torch.from_numpy(corr_xyz),
        }

        return sample_out

    def __len__(self):
        return len(self.data_dict['labels'])

    def get_class_indices(self, class_indices='all', asymmetric='True'):
        r"""Generate class indices.
        'all' -> all 40 classes.
        'seen' -> first 20 classes.
        'unseen' -> last 20 classes.
        list|tuple -> unchanged.
        asymmetric -> remove symmetric classes.
        """
        if isinstance(class_indices, str):
            assert class_indices in ['all', 'seen', 'unseen']
            if class_indices == 'all':
                class_indices = list(range(40))
            elif class_indices == 'seen':
                class_indices = list(range(20))
            else:
                class_indices = list(range(20, 40))
        if asymmetric:
            class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
        return class_indices


# class ModelNet40Cls(Dataset):
#     # fmt: off
#     ALL_CATEGORIES = [
#         'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
#         'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel',
#         'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool',
#         'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
#     ]
#
#     # fmt: on
#
#     def __init__(
#             self,
#             data_root: str,
#             split: str,
#             num_points: int = 2048,
#             use_normal: bool = True,
#             deterministic=False,
#             presample=True,
#             rotation_magnitude: float = 45.0,
#             translation_magnitude: float = 0.5,
#             noise_magnitude: Optional[float] = None,
#             keep_ratio: float = 0.7,
#             crop_method: str = 'plane',
#             twice_sample: bool = True,
#             # min_overlap: Optional[float] = None,
#             # max_overlap: Optional[float] = None,
#             class_indices: str = 'all',
#             asymmetric: bool = False,
#             try_count=30
#     ):
#         super(ModelNet40Cls, self).__init__()
#
#         assert split in ['train', 'trainval', 'val', 'test']
#         assert crop_method in ['plane', 'point']
#
#         self.split = split
#         self.num_points = num_points
#         self.deterministic = deterministic
#         self.presample = presample
#         self.rotation_magnitude = rotation_magnitude
#         self.translation_magnitude = translation_magnitude
#         self.noise_magnitude = noise_magnitude
#         self.keep_ratio = keep_ratio
#         self.crop_method = crop_method
#         self.asymmetric = asymmetric
#         self.class_indices = self.get_class_indices(class_indices, asymmetric)
#         self.use_normal = use_normal
#         self.twice_sample = twice_sample
#
#         # self.try_count = try_count
#         # self.min_overlap = min_overlap
#         # self.max_overlap = max_overlap
#         # self.twice_transform = twice_transform
#         # self.voxel_size = voxel_size
#         # self.overfitting_index = overfitting_index
#
#         if split in ['train', 'trainval']:
#             split = 'train'
#         else:
#             split = 'test'
#         pkl_name = os.path.join(data_root, f'modelnet40_cls_{split}.pkl')
#         if presample and not os.path.exists(pkl_name):
#             # store datas
#             with open(os.path.join(data_root, 'modelnet40_ply_hdf5_2048', f'{split}_files.txt')) as f:
#                 lines = f.readlines()
#             all_points = []
#             all_normals = []
#             all_labels = []
#             for line in lines:
#                 filename = line.strip()
#                 # h5file = h5py.File(f'modelnet40_ply_hdf5_2048/{filename}', 'r')
#                 h5file = h5py.File(filename, 'r')
#                 # for key in h5file.keys():     # keys: data, faceId, label, normal--error
#                 #     print(key)
#                 all_points.append(h5file['data'][:])
#                 all_labels.append(h5file['label'][:].flatten().astype(np.int32))
#             points = np.concatenate(all_points, axis=0)
#             print(f'calculating modelnet40_cls {split} normals, please waiting...')
#             for point in points:
#                 normal = estimate_normals(point)
#                 all_normals.append(normal)  # raw data: error normal
#             print(f'-------------------estimate modelnet40_cls {split} normals end--------------------------')
#             normals = np.array(all_normals)
#             labels = np.concatenate(all_labels, axis=0)
#
#             all_points.clear()
#             all_normals.clear()
#             all_labels.clear()
#             for i, label in enumerate(labels):
#                 if label in self.class_indices:
#                     all_points.append(points[i])
#                     all_normals.append(normals[i])
#                     all_labels.append(label)
#             points = np.array(all_points)  # train--(8284,2048,3)      test--(2148,2048,3)
#             normals = np.array(all_normals)  # train--(8284,2048,3)      test--(2148,2048,3)
#             labels = np.array(all_labels)  # train--(8284,)        test--(2148,)
#
#             # for label in labels:
#             #     if label in self.SYMMETRIC_CATEGORIES_ID:
#             #         print(label)
#
#             self.data_dict = {'points': points, 'normals': normals, 'labels': labels}
#             # save datas as pickle
#             dump_pickle(self.data_dict, pkl_name)
#             print(
#                 f"modelnet40_cls_{split}_pkl saved and load successfully. datas size is {len(self.data_dict['labels'])}")
#         else:
#             self.data_dict = load_pickle(pkl_name)
#             print(f"{osp.basename(pkl_name)} load successfully. datas size is {len(self.data_dict['labels'])}")
#
#     def __getitem__(self, index):
#         # if self.overfitting_index is not None:     # --PASS--
#         #     index = self.overfitting_index
#
#         # set deterministic
#         # if self.deterministic:  # --PASS--
#         #     np.random.seed(index)
#         raw_points = self.data_dict['points'][index]
#         raw_normals = self.data_dict['normals'][index]
#         label = self.data_dict['labels'][index]  # not using in point cloud register
#
#         points = normalize_points(raw_points)
#         normals = normalize_points(raw_normals)
#
#         if self.split in ['train', 'trainval']:
#             # random transform to source point cloud
#             transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
#             inv_transform = inverse_transform(transform)
#             points, normals = apply_transform(points, inv_transform, normals=normals)
#
#             # crop
#             if self.keep_ratio is not None:  # --Enter--
#                 if self.crop_method == 'plane':  # --Enter--     # 2048-->1434(2048*keep_ratio)
#                     points, normals = random_crop_point_cloud_with_plane(
#                         points, keep_ratio=self.keep_ratio, normals=normals)
#                 else:
#                     viewpoint = random_sample_viewpoint()
#                     points, normals = random_crop_point_cloud_with_point(
#                         points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=normals)
#             if self.twice_sample:  # --Enter--
#                 # twice sample on both point clouds.    # 1434-->1024
#                 points, normals = random_sample_points(points, self.num_points, normals=points)
#             # if self.voxel_size is not None:     # --PASS--
#             #     # voxel downsample reference point cloud
#             #     ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
#             #     src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)
#
#             # random jitter
#             if self.noise_magnitude is not None:  # --Enter--
#                 points = random_jitter_points(points, scale=0.01, noise_magnitude=self.noise_magnitude)
#             # random shuffle
#             '''may cause seg... are not correct'''
#             points, normals = random_shuffle_points(points, normals=normals)
#             points = points.astype(np.float32)
#             normals = normals.astype(np.float32)
#             # from utils.visualization import visualizepc2
#             # visualizepc1(points, self.ALL_CATEGORIES[int(label)])
#
#         else:
#             points, normals = fps_manual(torch.from_numpy(points).unsqueeze(0), self.num_points,
#                                          torch.from_numpy(normals).unsqueeze(0))
#             points, normals = points.squeeze(0).numpy(), normals.squeeze(0).numpy()
#
#         new_data_dict = {
#             # 'raw_points': raw_points.astype(np.float32),
#             'points': points.astype(np.float32),
#             'label': int(label)
#             # 'index': int(index),
#         }
#
#         if self.use_normal:  # --Enter--        # input_channel=6
#             # new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
#             new_data_dict['feats'] = np.concatenate((points, normals), axis=-1).astype(np.float32)
#         else:  # input_channel=3
#             new_data_dict['feats'] = points.astype(np.float32)
#             # visualizepc1(pts=points, str=self.ALL_CATEGORIES[int(label)])
#
#         signal_break = False
#         working_dir = os.getcwd()
#         debug_file = osp.join(working_dir, 'data_debugfile', 'modelnet_ref_pt_getitem')
#         if np.isnan(points).any():
#             signal_break = True
#             print(index)
#             np.save(osp.join(debug_file, points))
#             print('modelnet:   ' + 'points----nan--gititem()')
#         if np.isnan(normals).any():
#             signal_break = True
#             print(index)
#             np.save(osp.join(debug_file, normals))
#             print('modelnet:   ' + 'normals----nan--gititem()')
#         # print('modelnet40')
#         if signal_break:
#             exit(0)
#         return new_data_dict
#
#     def __len__(self):
#         return len(self.data_dict['labels'])
#         # return 3
#
#     def get_class_indices(self, class_indices='all', asymmetric='True'):
#         r"""Generate class indices.
#         'all' -> all 40 classes.
#         'seen' -> first 20 classes.
#         'unseen' -> last 20 classes.
#         list|tuple -> unchanged.
#         asymmetric -> remove symmetric classes.
#         """
#         if isinstance(class_indices, str):
#             assert class_indices in ['all', 'seen', 'unseen']
#             if class_indices == 'all':
#                 class_indices = list(range(40))
#             elif class_indices == 'seen':
#                 class_indices = list(range(20))
#             else:
#                 class_indices = list(range(20, 40))
#         if asymmetric:
#             class_indices = [x for x in class_indices if x in self.ASYMMETRIC_INDICES]
#         return class_indices

# class ModelNetHdf(Dataset):
#     def __init__(self, args, root: str, subset: str = 'train', categories: List = None, transform=None):
#         """ModelNet40 dataset from PointNet.
#         Automatically downloads the dataset if not available
#
#         Args:
#             root (str): Folder containing processed dataset
#             subset (str): Dataset subset, either 'train' or 'test'
#             categories (list): Categories to use
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.config = args
#         self._root = root
#         self.n_in_feats = args.in_feats_dim
#         self.overlap_radius = args.overlap_radius
#
#         if not os.path.exists(os.path.join(root)):
#             self._download_dataset(root)
#
#         with open(os.path.join(root, 'shape_names.txt')) as fid:
#             self._classes = [l.strip() for l in fid]
#             self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
#             self._idx2category = self._classes
#
#         with open(os.path.join(root, '{}_files.txt'.format(subset))) as fid:
#             h5_filelist = [line.strip() for line in fid]
#             h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
#             h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]
#
#         if categories is not None:
#             categories_idx = [self._category2idx[c] for c in categories]
#             self._classes = categories
#         else:
#             categories_idx = None
#
#         self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
#         self._transform = transform
#
#     def __getitem__(self, item):
#         sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}
#
#         # Apply perturbation
#         if self._transform:
#             sample = self._transform(sample)
#
#         corr_xyz = np.concatenate([  # concat src_corresp_pts&ref_corresp_pts:[012,345]
#             sample['points_src'][sample['correspondences'][0], :3],  # corresp-0:src, corresp-1:ref,
#             sample['points_ref'][sample['correspondences'][1], :3]], axis=1)  # [012]src&ref[345] corresp
#         # corr_xyz: shape(305(corresp_npts),6)
#
#         # Transform to my format
#         sample_out = {  # 无；overlap score
#             'src_xyz': torch.from_numpy(sample['points_src'][:, :3]),  # (717,3)
#             'tgt_xyz': torch.from_numpy(sample['points_ref'][:, :3]),  # (717,3)(pt)
#             'tgt_raw': torch.from_numpy(sample['points_raw'][:, :3]),  # (2048,3)(pt)
#             'src_overlap': torch.from_numpy(sample['src_overlap']),  # xxx_overlap表示该点是否在overlap中(717,)
#             'tgt_overlap': torch.from_numpy(sample['ref_overlap']),
#             'correspondences': torch.from_numpy(sample['correspondences']),
#             # correspondences表示src和ref对应的overlap-idx(2,corresp_num(305,267...))
#             'pose': torch.from_numpy(sample['transform_gt']),  # transform(3,4)
#             'idx': torch.from_numpy(sample['idx']),  # sample-id
#             'corr_xyz': torch.from_numpy(corr_xyz),  # (overlap_num, 6)(src_coor_pt, ref_coor_pt)
#         }
#
#         return sample_out
#
#     def __len__(self):
#         return self._data.shape[0]
#
#     @property
#     def classes(self):
#         return self._classes
#
#     @staticmethod
#     def _read_h5_files(fnames, categories):
#
#         all_data = []
#         all_labels = []
#
#         for fname in fnames:
#             f = h5py.File(fname, mode='r')
#             data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
#             labels = f['label'][:].flatten().astype(np.int64)
#
#             if categories is not None:  # Filter out unwanted categories
#                 mask = np.isin(labels, categories).flatten()
#                 data = data[mask, ...]
#                 labels = labels[mask, ...]
#
#             all_data.append(data)
#             all_labels.append(labels)
#
#         all_data = np.concatenate(all_data, axis=0)
#         all_labels = np.concatenate(all_labels, axis=0)
#         return all_data, all_labels
#
#     @staticmethod
#     def _download_dataset(root: str):
#         os.makedirs(root, exist_ok=True)
#
#         www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#         zipfile = os.path.basename(www)
#         os.system('wget {}'.format(www))
#         os.system('unzip {} -d .'.format(zipfile))
#         os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(root)))
#         os.system('rm {}'.format(zipfile))
#
#     def to_category(self, i):
#         return self._idx2category[i]
