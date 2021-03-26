import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl
from mmdet.datasets import TraumaDataset
from mmdet.datasets import build_dataloader
from torchvision import transforms as T
from pytorch_lightning.callbacks import ModelCheckpoint

# test model, a resnet 50

resnet = models.resnet50(pretrained=True)

# constants

BATCH_SIZE = 4
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 1
IMAGE_SIZE = 1280
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 4

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images)

    def training_step(self, batch, _):
        images = batch['img'].data[0].to(self.device)
        loss = self.forward(images)
        log = {'loss': loss}
        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

class NormalizeMeanVar(nn.Module):
    def __call__(self, img):
        return (img - img.mean([1, 2, 3], True)) / img.std([1, 2, 3], keepdim=True)

# main

if __name__ == '__main__':
    # dataset
    ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

    # minimalist pipeline
    pipeline = [
        dict(type='LoadImageFromFileUnchanged'),
        dict(type='RandomCrop', crop_size=(0.9, 0.9), crop_type='relative_range'),
        dict(type='Resize', img_scale=(1280, 1280), keep_ratio=False),
        dict(type='ReshapeChannels', one_channel=False),
        dict(type='AsType', new_type='np.float32'),
        # dict(type='NormalizeMinMax', minv=-1., maxv=1.),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img'], meta_keys=[])
    ]

    dataset = TraumaDataset(ann_file, pipeline)
    train_loader = build_dataloader(dataset,
                                    samples_per_gpu=4,
                                    workers_per_gpu=4,
                                    dist=False,
                                    shuffle=True)

    augment_pipeline = torch.nn.Sequential(
    #         T.RandomResizedCrop(size=1024, scale=(0.9, 1.0), ratio=(1., 1.)),
            # T.RandomApply(
            #     [T.ColorJitter(0.2, 0.2, 0.2, 0.2)],
            #     p = 0.3
            # ),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-20, 20), expand=False),
            NormalizeMeanVar()
    )

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        augment_fn=augment_pipeline,
        augment_fn2=augment_pipeline
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        save_top_k=30,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(
        gpus = [0],
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = False,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, train_loader)

