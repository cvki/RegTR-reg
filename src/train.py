import os, argparse

import yaml
from easydict import EasyDict

from cvhelpers.misc import prepare_logger
from cvhelpers.torch_helpers import setup_seed

from data_loaders import get_dataloader
from models import get_model
from trainer import Trainer
from utils.misc import load_config
from utils.utils import EasyConfig, make_log_dirs, Logger
from itertools import cycle

# setup_seed(0, cudnn_deterministic=False)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, help='Path to the config file.')
opt = parser.parse_args()       # get config path

if opt.cfg is None:
    print('Config not found')
    exit(-2)

cfg = EasyConfig()
cfg.load(opt.cfg)       # get config content to EasyConfig from cfg path

# create log dirs
make_log_dirs(cfg.logs)
# print(cfg)
logger = Logger(log_file=os.path.join(cfg.logs.log_dirs, 'logs_all.log'))
logger.debug(cfg)

def main():
    trainloader_modelnet, trainloader_shapenetpart, valloader_modelnet, valloader_shapenetpart = get_dataloader(cfg.dataset)
    train_loader=[ trainloader_modelnet, trainloader_shapenetpart]
    val_loader=[valloader_modelnet, valloader_shapenetpart]
    # train_loader = zip(cycle(trainloader_modelnet), trainloader_shapenetpart)
    # val_loader = zip(cycle(valloader_modelnet), valloader_shapenetpart)
    model = get_model(cfg)
    # trainer = Trainer(opt, niter=cfg.niter, grad_clip=cfg.grad_clip)
    trainer = Trainer(cfg.train_options)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
