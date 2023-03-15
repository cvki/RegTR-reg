import torch

import src.data_loaders.transforms
import src.data_loaders.modelnet as modelnet
import src.data_loaders.ShapeNetPart as shapenetpart
from src.data_loaders.collate_functions import collate_pair
from src.data_loaders.threedmatch import ThreeDMatchDataset

import torchvision


def get_dataloader(cfg):
    trainset_modelnet, valset_modelnet = modelnet.get_trainval_datasets(cfg)
    trainset_shapenetpart, valset_shapenetpart = shapenetpart.get_trainval_datasets(cfg)
    # test_dataset = modelnet.get_test_datasets(cfg)

    trainloader_modelnet = torch.utils.data.DataLoader(
        trainset_modelnet,
        batch_size=cfg.setting.train_batch_size,
        shuffle=True,
        num_workers=cfg.setting.num_workers,
        collate_fn=collate_pair,
    )

    valloader_modelnet = torch.utils.data.DataLoader(
        valset_modelnet,
        batch_size=cfg.setting.val_batch_size,
        shuffle=False,
        num_workers=cfg.setting.num_workers,
        collate_fn=collate_pair,
    )

    trainloader_shapenetpart = torch.utils.data.DataLoader(
        trainset_shapenetpart,
        batch_size=cfg.setting.train_batch_size,
        shuffle=True,
        num_workers=cfg.setting.num_workers,
        collate_fn=collate_pair,
    )

    valloader_shapenetpart = torch.utils.data.DataLoader(
        valset_shapenetpart,
        batch_size=cfg.setting.val_batch_size,
        shuffle=False,
        num_workers=cfg.setting.num_workers,
        collate_fn=collate_pair,
    )

   # test_loader=...
    return trainloader_modelnet, trainloader_shapenetpart, valloader_modelnet, valloader_shapenetpart
