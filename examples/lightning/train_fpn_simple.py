import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image
import pandas as pd

import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOLFPN
import pytorch_lightning as pl
from mmdet.datasets import TraumaDataset
from mmdet.datasets import build_dataloader
from mmcv.runner import load_checkpoint
from torchvision import transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint

import json
import cv2
import numpy as np
import itertools

from mmcv import Config
from mmdet.models import build_detector

# constants

BATCH_SIZE = 4
EPOCHS     = 100
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 1280
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 4

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOLFPN(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, _):
        loss = self.forward(batch)
        log = {'loss': loss}
        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)

        train_steps = (EPOCHS * len(self.train_dataloader.dataloader)) // self.trainer.accumulate_grad_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=1e-6)
        
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

class ImagesDataset(Dataset):
    def __init__(self, ann_file, image_size):
        super().__init__()
        with open(ann_file, 'r') as f:
            anns = json.load(f)

        self.paths = [ann['file_name'] for ann in anns['images']]
        self.transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.)),
#             T.RandomApply(
#                 [T.ColorJitter(0., 0.05, 0., 0.)],
#                 p = 0.8
#             ),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-20, 20), expand=False),
            NormalizeMeanVar()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path, -1).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)
        img = img.expand(3, -1, -1).unsqueeze(0)
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1.squeeze(), view2.squeeze()


class NormalizeMeanVar(nn.Module):
    def __call__(self, img):
        return (img - img.mean([1, 2, 3], True)) / img.std([1, 2, 3], keepdim=True)

class StackPooler(torch.nn.Module):
    def forward(self, x):
        return torch.stack([_x.mean([2, 3]) for _x in x]).transpose(0, 1)

# main

if __name__ == '__main__':
    # dataset
    ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

    df = pd.read_csv('/home/david/AZmed-ai/az_notebooks/exploration/samples_unique.csv', index_col=[0])

    dataset = ImagesDataset(ann_file, 1280)
    train_loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)

    cfg = Config.fromfile('/home/david/AZmed-ai/az_configs/trauma/retinanet_r50.py')
    model = build_detector(cfg.model)
    load_checkpoint(model, cfg.load_from, map_location='cpu')

    resnet = model.backbone
    fpn = model.neck

    # StackPooler does global average pooling on each FPN level and stacks outputs
    fpn_extractor = torch.nn.Sequential(
        resnet,
        fpn,
        StackPooler()
    ).to('cuda:2')

    model = SelfSupervisedLearner(
        fpn_extractor,
        image_size = IMAGE_SIZE,
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd() + '/fpn_simple',
        save_top_k=10,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(
        gpus = [2],
        max_epochs = EPOCHS,
        accumulate_grad_batches = 8,
        sync_batchnorm = False,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, train_loader)
