import os, sys, h5py, numpy as np, os.path as osp
from torch.utils.data import Dataset
from typing import Optional
from utils.utils import dump_pickle, load_pickle, normalize_points, random_sample_transform, apply_transform, \
    inverse_transform, \
    random_jitter_points, random_shuffle_points, estimate_normals, random_crop_point_cloud_with_plane, \
    random_crop_point_cloud_with_point, \
    compute_overlap, random_sample_viewpoint, random_sample_points
from .build import DATASETS
from utils.visualization import visualizepc1

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)


# class ScanObjectNNHardest(Dataset):
@DATASETS.register_module()
class ScanObjNNSmall(Dataset):
    """The hardest variant of ScanObjectNN.
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1],
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882.
    Args:
    """

    # can use all categories cause datas complicated
    ALL_CATEGORIES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                      "pillow", "sink", "sofa", "toilet"]  # 15
    ALL_CATEGORIES_ID = [id for id in
                         range(len(ALL_CATEGORIES))]  # 'cabinet, display, door' may can delete from asymmetric

    def __init__(self,
                 data_root: str,
                 split: str,
                 num_points: int = 2048,
                 use_normal=True,
                 deterministic=False,
                 rotation_magnitude: float = 45.0,
                 translation_magnitude: float = 0.5,
                 noise_magnitude: Optional[float] = None,
                 keep_ratio: float = 0.7,
                 crop_method: str = 'plane',
                 twice_sample: bool = True,
                 min_overlap: Optional[float] = None,
                 max_overlap: Optional[float] = None,
                 try_count=30,
                 presample=False,
                 hardest=False
                 ):
        super().__init__()
        self.split = split
        self.use_normal = use_normal
        self.num_points = num_points
        self.presample = presample
        self.deterministic = deterministic
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.twice_sample = twice_sample
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.hardest = hardest
        self.try_count = try_count
        # self.asymmetric = asymmetric,
        # self.class_indices = class_indices,

        if split in ['train', 'trainval']:
            split = 'train'
        else:
            split = 'test'

        # if self.hardest:
        #     data_root=os.path.join(data_root, 'scanobjnn_hardest')
        #     h5_name = os.path.join(data_root, f'{split}_objectdataset_augmentedrot_scale75.h5')
        #     # dict_key(shape): data(11416,2048,3),label(11416,),mask(11416,2048); 2882
        # else:
        #     data_root = os.path.join(data_root, 'scanobjnn_small')
        #     h5_name = os.path.join(data_root, f'{split}.h5')  # 2309, 581

        data_root = os.path.join(data_root, 'scanobjnn_small')
        # train--2309, test--581
        h5_name = os.path.join(data_root, f'{split}.h5')
        pkl_name = os.path.join(data_root, 'scanObjnn_{}.pkl'.format(split))

        # with h5py.File(h5_name, 'r') as f:
        #     print(f.keys())     # <KeysViewHDF5 ['data', 'label', 'mask']>

        '''This method only split train-set and val-set, and the test-set is from val-set of random cutting a half'''
        if presample and not os.path.exists(pkl_name):
            if not osp.isfile(h5_name):
                raise FileExistsError(
                    f'{h5_name} does not exist, please download dataset at first')
            with h5py.File(h5_name, 'r') as f:
                # for key in f.keys():    # keys: data, label, mask
                #     print(key)
                #     points = f['data']      # small: train--(2309,2048,3)  test--581         large:(train--11416, test--2882)
                #     labels = f['label']     # small: train--(2309,)  test--581         large:(train--11416, test--2882)
                #     mask = f['mask']        # small: train--(2309,2048)  test--581      large:(train--11416, test--2882)   # mask of every point for seg
                # # show some object of labels
                # count_upline=10
                # for i in self.ALL_CATEGORIES_ID:
                #     count=0
                #     for id, label in enumerate(labels):
                #         if label==self.ALL_CATEGORIES_ID[i]:
                #             count+=1
                #             visualizepc1(points[id], self.ALL_CATEGORIES[label])
                #             if count>count_upline:
                #                 break
                self.points = np.array(f['data']).astype(np.float32)
                normals = []
                print(f'calculating scanobjnn {split} normals, please waiting...')
                for one_points in self.points:
                    # 计算法向量比较慢，这个时候可以将算出的normal和points和label保存到pickle中
                    normals.append(estimate_normals(one_points).astype(np.float32))
                print(f'-------------------estimate {split} normals end--------------------------')
                self.normals = np.array(normals).astype(np.float32)
                self.cls = np.array(f['label']).astype(int)  # cls for classification
                # print(os.getcwd())
                dump_pickle({'points': self.points,
                             'normals': self.normals,
                             'cls': self.cls},
                            osp.join(data_root, 'scanObjnn_{}.pkl'.format(split)))  # change it to your storing path
        else:
            datas = load_pickle(pkl_name)
            self.points, self.normals, self.cls = datas['points'], datas['normals'], datas['cls']
        print(f'ScanObjNNSmall_{split} load successfully. datas size is {len(np.squeeze(self.cls))}')

    # @property
    # def num_classes(self):
    #     return self.labels.max() + 1

    def __getitem__(self, idx):
        # if self.overfitting_index is not None:     # --PASS--
        #     index = self.overfitting_index

        # set deterministic
        # if self.deterministic:  # --Enter--
        #     np.random.seed(idx)
        raw_points = self.points[idx, :, :]
        # visualizepc1(raw_points)
        raw_normals = self.normals[idx, :, :]
        ref_points = normalize_points(np.squeeze(raw_points))
        # add--
        ref_normals = normalize_points(np.squeeze(raw_normals))
        if self.split in ['train', 'trainval']:
            src_points = ref_points.copy()
            src_normals = ref_normals.copy()
            cls = self.cls[idx]  # not using in point cloud register

            # random transform to source point cloud
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform = inverse_transform(transform)
            src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

            raw_ref_points = ref_points
            raw_ref_normals = ref_normals
            raw_src_points = src_points
            raw_src_normals = src_normals
            try_count = self.try_count,
            while try_count > 0:
                ref_points = raw_ref_points
                ref_normals = raw_ref_normals
                src_points = raw_src_points
                src_normals = raw_src_normals
                # crop
                if self.keep_ratio is not None:  # --Enter--
                    if self.crop_method == 'plane':  # --Enter--     # 2048-->1434(2048*keep_ratio)
                        ref_points, ref_normals = random_crop_point_cloud_with_plane(
                            ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_plane(
                            src_points, keep_ratio=self.keep_ratio, normals=src_normals
                        )
                    else:
                        viewpoint = random_sample_viewpoint()
                        ref_points, ref_normals = random_crop_point_cloud_with_point(
                            ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_point(
                            src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                        )

                # data check
                is_available = True
                # check overlap
                if self.check_overlap:  # --PASS--
                    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                    if self.min_overlap is not None:
                        is_available = is_available and overlap >= self.min_overlap
                    if self.max_overlap is not None:
                        is_available = is_available and overlap <= self.max_overlap
                try_count -= 1
                if is_available:  # --Enter--
                    break

            if self.twice_sample:  # --Enter--
                # twice sample on both point clouds.    # 1434-->1024
                ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
                src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

            # if self.voxel_size is not None:     # --PASS--
            #     # voxel downsample reference point cloud
            #     ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            #     src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

            # random jitter
            if self.noise_magnitude is not None:  # --Enter--
                ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
                src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

            # random shuffle
            '''may cause cls ,seg... are not correct'''
            ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
            src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        else:
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points = ref_points.copy()
            src_normals = ref_normals.copy()
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform = inverse_transform(transform)
            src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        new_data_dict = {
            # 'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            # 'seg': seg,
            # 'cls':cls,
            # 'index': int(index),
        }

        if self.use_normal:  # --Enter--        # input_channel=6
            # new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            # ref_normals = ref_normals.astype(np.float32)
            # src_normals = src_normals.astype(np.float32)
            new_data_dict['ref_feats'] = np.concatenate((ref_points, ref_normals), axis=-1).astype(np.float32)
            new_data_dict['src_feats'] = np.concatenate((src_points, src_normals), axis=-1).astype(np.float32)
        else:  # input_channel=3
            new_data_dict['ref_feats'] = ref_points
            new_data_dict['src_feats'] = src_points

        return new_data_dict

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """


@DATASETS.register_module()
class ScanObjNNHardest(Dataset):
    """The hardest variant of ScanObjectNN.
    The data we use is: `training_objectdataset_augmentedrot_scale75.h5`[1],
    where there are 2048 points in training and testing.
    The number of training samples is: 11416, and the number of testing samples is 2882.
    Args:
    """

    # can use all categories cause datas complicated
    ALL_CATEGORIES = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                      "pillow", "sink", "sofa", "toilet"]  # 15
    ALL_CATEGORIES_ID = [id for id in
                         range(len(ALL_CATEGORIES))]  # 'cabinet, display, door' may can delete from asymmetric

    def __init__(self,
                 data_root: str,
                 split: str,
                 num_points: int = 2048,
                 use_normal=True,
                 deterministic=False,
                 rotation_magnitude: float = 45.0,
                 translation_magnitude: float = 0.5,
                 noise_magnitude: Optional[float] = None,
                 keep_ratio: float = 0.7,
                 crop_method: str = 'plane',
                 twice_sample: bool = True,
                 min_overlap: Optional[float] = None,
                 max_overlap: Optional[float] = None,
                 try_count=30,
                 presample=False,
                 hardest=False
                 ):
        super().__init__()
        self.split = split
        self.use_normal = use_normal
        self.num_points = num_points
        self.presample = presample
        self.deterministic = deterministic
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.twice_sample = twice_sample
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.hardest = hardest
        self.try_count = try_count
        # self.asymmetric = asymmetric,
        # self.class_indices = class_indices,

        if split in ['train', 'trainval']:
            split = 'train'
        else:
            split = 'test'

        # if self.hardest:
        #     data_root=os.path.join(data_root, 'scanobjnn_hardest')
        #     h5_name = os.path.join(data_root, f'{split}_objectdataset_augmentedrot_scale75.h5')
        #     # dict_key(shape): data(11416,2048,3),label(11416,),mask(11416,2048); 2882
        # else:
        # data_root = os.path.join(data_root, 'scanobjnn_hardest')
        # h5_name = os.path.join(data_root, f'{split}.h5')  # 2309, 581
        # pkl_name = os.path.join(data_root, 'scanObjnn_{}.pkl'.format(split))

        data_root = os.path.join(data_root, 'scanobjnn_hardest')
        h5_name = os.path.join(data_root, f'{split}_objectdataset_augmentedrot_scale75.h5')
        pkl_name = os.path.join(data_root, 'scanObjnn_{}.pkl'.format(split))
        # dict_key(shape): data(11416,2048,3),label(11416,),mask(11416,2048); 2882

        # with h5py.File(h5_name, 'r') as f:
        #     print(f.keys())     # <KeysViewHDF5 ['data', 'label', 'mask']>

        '''This method only split train-set and val-set, and the test-set is from val-set of random cutting a half'''
        if presample and not os.path.exists(pkl_name):
            if not osp.isfile(h5_name):
                raise FileExistsError(
                    f'{h5_name} does not exist, please download dataset firstly')
            with h5py.File(h5_name, 'r') as f:
                # for key in f.keys():    # keys: data, label, mask
                #     print(key)
                #     points = f['data']      # small: train--(2309,2048,3)  test--581         large:(train--11416, test--2882)
                #     labels = f['label']     # small: train--(2309,)  test--581         large:(train--11416, test--2882)
                #     mask = f['mask']        # small: train--(2309,2048)  test--581      large:(train--11416, test--2882)   # mask of every point for seg
                # # show some object of labels
                # count_upline=10
                # for i in self.ALL_CATEGORIES_ID:
                #     count=0
                #     for id, label in enumerate(labels):
                #         if label==self.ALL_CATEGORIES_ID[i]:
                #             count+=1
                #             visualizepc1(points[id], self.ALL_CATEGORIES[label])
                #             if count>count_upline:
                #                 break
                self.points = np.array(f['data']).astype(np.float32)
                normals = []
                print(f'calculating scanobjnn {split} normals, please waiting...')
                for one_points in self.points:
                    # 计算法向量比较慢，这个时候可以将算出的normal和points和label保存到pickle中
                    normals.append(estimate_normals(one_points).astype(np.float32))
                print(f'-------------------estimate {split} normals end--------------------------')
                self.normals = np.array(normals).astype(np.float32)
                self.cls = np.array(f['label']).astype(int)  # cls for classification
                print(os.getcwd())
                dump_pickle({'points': self.points,
                             'normals': self.normals,
                             'cls': self.cls},
                            osp.join(data_root, 'scanObjnn_{}.pkl'.format(split)))  # change it to your storing path
        else:
            datas = load_pickle(pkl_name)
            self.points, self.normals, self.cls = datas['points'], datas['normals'], datas['cls']
        print(f'ScanObjNNHardest_{split} load successfully. datas size is {len(np.squeeze(self.cls))}')

    # @property
    # def num_classes(self):
    #     return self.labels.max() + 1

    def __getitem__(self, idx):
        # if self.overfitting_index is not None:     # --PASS--
        #     index = self.overfitting_index

        # set deterministic
        if self.deterministic:  # --PASS--
            np.random.seed(idx)
        raw_points = self.points[idx, :, :]
        # visualizepc1(raw_points)
        raw_normals = self.normals[idx, :, :]
        ref_points = normalize_points(np.squeeze(raw_points))
        # add--
        ref_normals = normalize_points(np.squeeze(raw_normals))
        if self.split in ['train', 'trainval']:
            src_points = ref_points.copy()
            src_normals = ref_normals.copy()
            cls = self.cls[idx]  # not using in point cloud register

            # random transform to source point cloud
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform = inverse_transform(transform)
            src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

            raw_ref_points = ref_points
            raw_ref_normals = ref_normals
            raw_src_points = src_points
            raw_src_normals = src_normals
            try_count = self.try_count,
            while try_count > 0:
                ref_points = raw_ref_points
                ref_normals = raw_ref_normals
                src_points = raw_src_points
                src_normals = raw_src_normals
                # crop
                if self.keep_ratio is not None:  # --Enter--
                    if self.crop_method == 'plane':  # --Enter--     # 2048-->1434(2048*keep_ratio)
                        ref_points, ref_normals = random_crop_point_cloud_with_plane(
                            ref_points, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_plane(
                            src_points, keep_ratio=self.keep_ratio, normals=src_normals
                        )
                    else:
                        viewpoint = random_sample_viewpoint()
                        ref_points, ref_normals = random_crop_point_cloud_with_point(
                            ref_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=ref_normals
                        )
                        src_points, src_normals = random_crop_point_cloud_with_point(
                            src_points, viewpoint=viewpoint, keep_ratio=self.keep_ratio, normals=src_normals
                        )

                # data check
                is_available = True
                # check overlap
                if self.check_overlap:  # --PASS--
                    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=0.05)
                    if self.min_overlap is not None:
                        is_available = is_available and overlap >= self.min_overlap
                    if self.max_overlap is not None:
                        is_available = is_available and overlap <= self.max_overlap
                try_count -= 1
                if is_available:  # --Enter--
                    break

            if self.twice_sample:  # --Enter--
                # twice sample on both point clouds.    # 1434-->1024
                ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
                src_points, src_normals = random_sample_points(src_points, self.num_points, normals=src_normals)

            # if self.voxel_size is not None:     # --PASS--
            #     # voxel downsample reference point cloud
            #     ref_points, ref_normals = voxel_downsample(ref_points, self.voxel_size, normals=ref_normals)
            #     src_points, src_normals = voxel_downsample(src_points, self.voxel_size, normals=src_normals)

            # random jitter
            if self.noise_magnitude is not None:  # --Enter--
                ref_points = random_jitter_points(ref_points, scale=0.01, noise_magnitude=self.noise_magnitude)
                src_points = random_jitter_points(src_points, scale=0.01, noise_magnitude=self.noise_magnitude)

            # random shuffle
            '''may cause cls ,seg... are not correct'''
            ref_points, ref_normals = random_shuffle_points(ref_points, normals=ref_normals)
            src_points, src_normals = random_shuffle_points(src_points, normals=src_normals)

        else:
            ref_points, ref_normals = random_sample_points(ref_points, self.num_points, normals=ref_normals)
            src_points = ref_points.copy()
            src_normals = ref_normals.copy()
            transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
            inv_transform = inverse_transform(transform)
            src_points, src_normals = apply_transform(src_points, inv_transform, normals=src_normals)

        new_data_dict = {
            # 'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            # 'seg': seg,
            # 'cls':cls,
            # 'index': int(index),
        }

        if self.use_normal:  # --Enter--        # input_channel=6
            # new_data_dict['raw_normals'] = raw_normals.astype(np.float32)
            # ref_normals = ref_normals.astype(np.float32)
            # src_normals = src_normals.astype(np.float32)
            new_data_dict['ref_feats'] = np.concatenate((ref_points, ref_normals), axis=-1).astype(np.float32)
            new_data_dict['src_feats'] = np.concatenate((src_points, src_normals), axis=-1).astype(np.float32)
        else:  # input_channel=3
            new_data_dict['ref_feats'] = ref_points
            new_data_dict['src_feats'] = src_points

        return new_data_dict

    def __len__(self):
        return self.points.shape[0]

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """


if __name__ == '__main__':
    # small_data_root = '/d2/code/gitRes/Expts/local_rgstr_clsSeg/datas/ScanObjNN/scanobjnn_small'  # contain: 'train.h5', 'test.h5'
    # large_data_root = '/d2/code/gitRes/Expts/local_rgstr_clsSeg/datas/ScanObjNN/scanobjnn_hardest'  # contain: 'train.h5', 'test.h5'
    # signal_dataset = input('select datasets idx(input 1 or 2):   1.large_scanobjnn,  2.small_scanobjnn')
    # if signal_dataset == str(1):
    #     trainset_scanobjnn = ScanObjNN(data_root=large_data_root, split='train', num_points=1024, presample=True,
    #                                    hardest=True)
    #     testset_scanobjnn = ScanObjNN(data_root=large_data_root, split='test', num_points=1024, presample=True,
    #                                   hardest=True)
    # elif signal_dataset == str(2):
    #     trainset_scanobjnn = ScanObjNN(data_root=small_data_root, split='train', num_points=1024, presample=True)
    #     testset_scanobjnn = ScanObjNN(data_root=small_data_root, split='test', num_points=1024, presample=True)
    # else:
    #     print('input  ERROR!')
    #     exit(0)
    # from torch.utils.data import DataLoader
    #
    # scanobjnn_trainloader = DataLoader(dataset=trainset_scanobjnn, batch_size=4, shuffle=True, drop_last=True)
    # scanobjnn_testloader = DataLoader(dataset=testset_scanobjnn, batch_size=4, shuffle=True, drop_last=False)
    # for data_dict in scanobjnn_trainloader:
    #     print("--------------------------------test loader---------------------------------")
    pass

