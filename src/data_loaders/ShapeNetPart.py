import argparse
import os
import glob
import h5py
import json
import pickle
import logging

import torchvision
from tqdm import tqdm
import numpy as np
import torch
from typing import Optional, List
from torch.utils.data import Dataset
from . import modelnet_transforms as Transforms

from src.utils.utils import fps_manual, normalize_points, random_sample_transform, apply_transform, inverse_transform, \
    random_jitter_points, random_shuffle_points, estimate_normals, random_crop_point_cloud_with_plane, \
    random_sample_viewpoint, random_crop_point_cloud_with_point, compute_overlap, random_sample_points, \
    regularize_normals

from src.utils.viz import visualizepc1

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
ALL_CATEGORIES = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop',
                  'Motorbike',
                  'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']  # 16
SYMMETRIC_CATEGORIES = ['Lamp', 'Mug', 'Rocket']
SYMMETRIC_CATEGORIES_ID = [8, 11, 13]
ASYMMETRIC_CATEGORIES = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Laptop', 'Motorbike',
                         'Pistol', 'Skateboard', 'Table']
ASYMMETRIC_CATEGORIES_ID = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15]


def download_shapenetpart(DATA_DIR):
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR)):
        os.mkdir(os.path.join(DATA_DIR))
        www = 'https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], os.path.join(DATA_DIR)))
        os.system('rm %s' % (zipfile))


def get_class_indices(self, class_indices, asymmetric):
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


def load_data_partseg(DATA_DIR, partition):
    # download_shapenetpart(DATA_DIR)
    tmp_data = []
    tmp_label = []
    tmp_seg = []
    all_data = []
    all_label = []
    all_seg = []
    all_normal = []
    if partition in ['trainval', 'train']:
        file = glob.glob(os.path.join(DATA_DIR, '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, '*%s*.h5' % partition))
    for h5_name in file:
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            # for key in f.keys():        # data, label, pid
            #     print(f[key].name)
        tmp_data.append(data)
        tmp_label.append(label)
        tmp_seg.append(seg)
    tmp_data = np.concatenate(tmp_data, axis=0)  # (14007, 2048, 3)
    tmp_label = np.concatenate(tmp_label, axis=0)  # (14007, 1)
    tmp_seg = np.concatenate(tmp_seg, axis=0)  # (14007, 2048)
    for i, label in tqdm(enumerate(tmp_label)):
        if label not in SYMMETRIC_CATEGORIES_ID:
            all_data.append(tmp_data[i, :, :])
            all_label.append(tmp_label[i, :])
            all_seg.append(tmp_seg[i, :])
            # estimate_normals
            all_normal.append(estimate_normals(tmp_data[i, :, :]))

    print(np.shape(all_data))  # (12746, 2048, 3)
    print(np.shape(all_normal))  # (12746, 2048, 3)
    print(np.shape(all_label))  # (12746, 1)
    print(np.shape(all_seg))  # (12746, 2048)

    return all_data, all_normal, all_label, all_seg


class ShapeNetPart(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car', 'chair',
               'earphone', 'guitar', 'knife', 'lamp', 'laptop',
               'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]

    cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                 'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                 'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                 'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
    cls2parts = []
    cls2partembed = torch.zeros(16, 50)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for lbl in cls_parts[cat]:
            part2cls[lbl] = cat

    def __init__(self, transform=None, args=None):
        # self.args=args
        catfile = os.path.join(args.shapenetpart.data_root, 'synsetoffset2category.txt')
        self.data_root = args.shapenetpart.data_root
        self.num_points = args.setting.num_points
        self.use_normal = args.setting.use_normal
        self.presample = args.setting.presample  # if you first run it, please set it 'True' to generate dataset.pkl
        self.asymmetric = args.shapenetpart.asymmetric      # if False: all datas-13998,2874
        self.split = args.setting.split
        self.transform = transform
        # self.part_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

        if self.split in ['train', 'trainval']:
            split = 'train'
        else:
            split = 'test'
        if self.asymmetric:
            class_choice = ASYMMETRIC_CATEGORIES
            filename = os.path.join(self.data_root, 'processed',
                                    f'shapeNetPart_reg_{split}_normal{int(self.use_normal)}.pkl')
        else:
            class_choice = ALL_CATEGORIES
            filename = os.path.join(self.data_root, 'processed',
                                    f'shapeNetPart_reg_symm_{split}_normal{int(self.use_normal)}.pkl')


        if self.presample and not os.path.exists(filename):
            cat_tmp = {}
            with open(catfile, 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    cat_tmp[ls[0]] = ls[1]
            cat_tmp = {k: v for k, v in cat_tmp.items()}  # 16item-dict :{label:filename}
            classes_original = dict(zip(cat_tmp, range(len(cat_tmp))))  # # 16item-dict :{label_name:label_id}, str2int

            cat = {}
            if not class_choice is None:  # use labels to select obj(some object may dont need)
                for k, v in cat_tmp.items():
                    if k in class_choice:
                        cat[k] = v

            del cat_tmp

            meta = {}
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
                test_ids = set([str(d.split('/')[2]) for d in json.load(f)])

            for item in cat:
                meta[item] = []
                dir_point = os.path.join(self.data_root, cat[item])
                fns = sorted(os.listdir(dir_point))
                if self.split in ['trainval', 'train']:
                    fns = [fn for fn in fns if (
                            (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
                # elif self.split == 'train':
                #     fns = [fn for fn in fns if fn[0:-4] in train_ids]
                # elif self.split == 'val':
                #     fns = [fn for fn in fns if fn[0:-4] in val_ids]
                elif self.split in ['val', 'test']:
                    fns = [fn for fn in fns if fn[0:-4] in test_ids]
                else:
                    print('Unknown split: %s. Exiting..' % (self.split))
                    exit(-1)
                for fn in fns:
                    token = (os.path.splitext(os.path.basename(fn))[0])
                    meta[item].append(os.path.join(dir_point, token + '.txt'))

            datapath = []
            for item in cat:
                for fn in meta[item]:
                    datapath.append((item, fn))  # datas path: (12537，)

            classes = {}
            for i in cat.keys():
                classes[i] = classes_original[i]

            # if transform is None:
            #     self.eye = np.eye(shape_classes)
            # else:
            #     self.eye = torch.eye(shape_classes)

            # create pickle datas
            np.random.seed(0)
            self.points, self.normals, self.rgb, self.labels, self.cls = [], [], [], [], []  # cls--大类用于分类   labels--小类用于分割
            npoints = []
            if self.use_normal:
                print(f'calculating shapenetpart {split} normals, please waiting...')
            for cat, filepath in tqdm(datapath, desc=f'package ShapeNetPart {split} split'):
                cls = classes[cat]
                # cls = np.array([cls]).astype(np.int64)
                self.cls.append(np.array(cls).astype(np.int64))  # (1,)
                datas = np.loadtxt(filepath).astype(np.float32)
                self.points.append(datas[:, 0:3])  # (N,3)
                self.rgb.append(datas[:, 3:6])  # (N,3)
                self.labels.append(datas[:, -1].astype(np.int64))  # (N,)
                npoints.append(len(datas))
                # calculate normals
                if self.use_normal:
                    normal = estimate_normals(datas[:, 0:3]).astype(np.float32)
                    self.normals.append(normal)
            if self.use_normal:
                print(f'-------------------estimate {split} normals end--------------------------')

            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(os.path.join(self.data_root, 'processed'), exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump({'points': self.points,
                             'normals': self.normals,
                             'rgb': self.rgb,
                             'labels': self.labels,
                             'cls': self.cls}, f)
                print(f"{os.path.basename(filename)} saved successfully")
        else:
            with open(filename, 'rb') as f:
                datas = pickle.load(f)
                # self.points, self.normals, self.rgb, self.labels, self.cls = datas['points'],datas['normals'],datas['rgb'],datas['labels'],datas['cls']
                self.points, self.normals, self.cls = datas['points'], datas['normals'], datas['cls']
                print(f"{os.path.basename(filename)} load successfully. datas size is {len(np.squeeze(self.cls))}")

    def __getitem__(self, item):
        sample = {
            'points': np.concatenate([self.points[item], self.normals[item]], axis=-1)
            if self.use_normal else self.points[item],
            'label': self.cls[item], 'idx': np.array(item, dtype=np.int32)}

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

    # def __getitem__(self, index):
    #     # if self.overfitting_index is not None:     # --PASS--
    #     #     index = self.overfitting_index
    #     # # set deterministic
    #     if self.deterministic:  # --PASS--
    #         np.random.seed(index)
    #
    #     raw_points = self.points[index]
    #     seg = self.labels[index]        # not using in point cloud register
    #     cls=self.cls[index]         # not using in point cloud register
    #     if self.use_normal:
    #         raw_normals = self.normals[index]
    #     else:
    #         raw_normals=self.rgb[index]     # use rgb or [if only use points,then change getitem by removing the key of 'normals']
    #     # normalize point cloud
    #     ref_points = normalize_points(raw_points)
    #     ref_normals = normalize_points(raw_normals)
    #     if self.split in ['train', 'trainval']:
    #         src_points = ref_points.copy()
    #         src_normals = ref_normals.copy()
    #
    #         # random transform to source point cloud
    #         transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
    #         inv_transform = inverse_transform(transform)
    #         src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)
    #
    #         raw_ref_points = ref_points
    #         raw_ref_normals = ref_normals
    #         raw_src_points = src_points
    #         raw_src_normals = src_normals
    #         try_count=self.try_count
    #         while try_count>0:
    #             ref_points = raw_ref_points
    #             ref_normals = raw_ref_normals
    #             src_points = raw_src_points
    #             src_normals = raw_src_normals
    #             # crop
    #             if self.keep_ratio is not None:  # --Enter--
    #                 if self.crop_method == 'plane':  # --Enter--     # aaaa-->xxxx(aaaa*keep_ratio) 每个点云点的数目不一致
    #                     ref_points, ref_normals = random_crop_point_cloud_with_plane(
    #                         ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
    #                     )
    #                     src_points, src_normals = random_crop_point_cloud_with_plane(
    #                         src_points, keep_ratio=self.keep_ratio, normals=src_normals
    #                     )
    #                 else:
    #                     viewpoint = random_sample_viewpoint()
    #                     ref_points, ref_normals = random_crop_point_cloud_with_point(
    #                         ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
    #                     )
    #                     src_points, src_normals = random_crop_point_cloud_with_point(
    #                         src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
    #                     )
    #             # data check
    #             is_available = True
    #             # check overlap
    #             if self.check_overlap:  # --PASS--
    #                 overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
    #                 if self.min_overlap is not None:
    #                     is_available = is_available and overlap >= self.min_overlap
    #                 if self.max_overlap is not None:
    #                     is_available = is_available and overlap <= self.max_overlap
    #             try_count-=1
    #             if is_available:  # --Enter--
    #                 break
    #
    #         if self.twice_sample:  # --Enter--
    #             # twice sample on both point clouds.    # xxxx-->1024
    #             ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
    #             src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)
    #
    #         # if self.voxel_size is not None:     # --PASS--
    #         #     # voxel downsample reference point cloud
    #         #     ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
    #         #     src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)
    #
    #         # random jitter
    #         if self.noise_magnitude is not None:  # --Enter--
    #             ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
    #             src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)
    #
    #         # random shuffle
    #         '''may cause cls ,seg... are not correct'''
    #         ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
    #         src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)
    #
    #     # else:
    #     #     ref_points = raw_points       # raw中每个点云点的数目不一致，因此如果在测试中batchsize不为1时，使用raw数据则报错，除非重写collate_fn
    #     #     src_points = raw_points
    #     #     ref_normals = raw_normals
    #     #     src_normals = raw_normals
    #
    #     else:       # shapernetpart中每个点云的点的数目不同，这里进行随即裁剪
    #         ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
    #         src_points = ref_points.copy()
    #         src_normals = ref_normals.copy()
    #         transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
    #         inv_transform = inverse_transform(transform)
    #         src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)
    #
    #     new_data_dict = {
    #         # 'raw_points': raw_points.astype(np.float32),      # collate_fn error because raw_points shapes are not equal,need rewrite collate_fn
    #         'ref_points': ref_points.astype(np.float32),
    #         'src_points': src_points.astype(np.float32),
    #         'transform': transform.astype(np.float32),
    #         # 'seg': seg,
    #         # 'cls':cls,
    #         # 'index': int(index),
    #     }
    #
    #     if self.use_normal:  # --Enter--        # input_channel=6
    #         # new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
    #         # ref_normals = ref_normals.astype(np.float32)
    #         # src_normals = src_normals.astype(np.float32)
    #         new_data_dict['ref_feats'] = np.concatenate((ref_points, ref_normals), axis=-1).astype(np.float32)
    #         new_data_dict['src_feats'] = np.concatenate((src_points, src_normals), axis=-1).astype(np.float32)
    #     else:  # input_channel=3
    #         new_data_dict['ref_feats'] = ref_points
    #         new_data_dict['src_feats'] = src_points
    #
    #     signal_break = False
    #     working_dir = os.getcwd()
    #     debug_file = os.path.join(working_dir, 'data_debugfile', 'shapepart_ref_pt_getitem')
    #     if np.isnan(ref_points).any():
    #         signal_break = True
    #         print(index)
    #         np.save(os.path.join(debug_file, ref_points))
    #         print('shapenetpart:   '+ 'ref_points----nan--gititem()')
    #     if np.isnan(ref_normals).any():
    #         signal_break = True
    #         print(index)
    #         np.save(os.path.join(debug_file, ref_normals))
    #         print('shapenetpart:   '+ 'ref_normals----nan--gititem()')
    #     if np.isnan(src_points).any():
    #         signal_break = True
    #         print(index)
    #         np.save(os.path.join(debug_file, src_points))
    #         print('shapenetpart:   '+ 'src_points----nan--gititem()')
    #     if np.isnan(src_normals).any():
    #         signal_break = True
    #         print(index)
    #         np.save(os.path.join(debug_file, src_normals))
    #         print('shapenetpart:   '+ 'src_normals----nan--gititem()')
    #     if signal_break is True:
    #         print('---------end--------------')
    #         exit(0)
    #
    #     # print('shapenetpart')
    #     return new_data_dict

    def __len__(self):
        return len(self.cls)
        # return 7


def get_trainval_datasets(args: argparse.Namespace):
    # train_categories, val_categories = None, None
    # if args.train_categoryfile:
    #     train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
    #     train_categories.sort()
    # if args.val_categoryfile:
    #     val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
    #     val_categories.sort()

    '''this transform is same as modelnet's'''
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
    train_dataset = ShapeNetPart(train_transforms, args)

    args.setting.update(dict(split='val'))
    val_dataset = ShapeNetPart(val_transforms, args)

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
        train_transforms = [Transforms.SplitSourceRef(),
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


if __name__ == '__main__':
    # labels = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike',
    #           'mug', 'pistol',  'rocket', 'skateboard', 'table']
    # labels_id=[i for i in range(len(labels))]
    # test_datapath = '/d2/code/datas/shapenet_part_seg_hdf5_data/hdf5_data'
    #
    # point,normal,label,seg=load_data_partseg(test_datapath, 'train')
    # for data,label in zip(data,label):
    #     if label[0]==8:
    #         visualizepc1(data,labels[label[0]])
    # load_data_partseg(test_datapath, 'test')
    # print('---------------end----------------')
    shapenetpart_trainset = ShapeNetPart(presample=False, num_points=1024, split='trainval')
    shapenetpart_testset = ShapeNetPart(presample=False, num_points=1024, split='test')

    for i, data in enumerate(shapenetpart_trainset.points):
        if np.isnan(data).any():
            print(i)
            print('shapenetpart_trainset_points_nan')

    for i, data in enumerate(shapenetpart_trainset.normals):
        if np.isnan(data).any():
            print(i)
            print('shapenetpart_trainset_normals_nan')

    print('---------end--------------')

    # from torch.utils.data import DataLoader
    # shapenetpart_trainloader=DataLoader(dataset=shapenetpart_trainset,batch_size=4,shuffle=True,drop_last=True)
    # # shapenetpart_testloader=DataLoader(dataset=shapenetpart_testset,batch_size=4,shuffle=True)
    # for datas in shapenetpart_trainloader:
    #     print('----------------test dataloader----------------')
