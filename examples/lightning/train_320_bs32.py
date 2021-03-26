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
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets import ImagesDataset
from transforms import pipeline_jitter, pipeline_nojitter
import json
import cv2
import numpy as np

# test model, a resnet 50


# constants

BATCH_SIZE = 32
EPOCHS     = 100
LR         = 3e-4
# LR         = 1e-4
NUM_GPUS   = 1
IMAGE_SIZE = 320
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
        loss = self.forward(batch)
        log = {'loss': loss}
        return {'loss': loss, 'log': log}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
        
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()


# main
if __name__ == '__main__':
    # dataset
    ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

    # minimalist pipeline
    transform = pipeline_nojitter(image_size=IMAGE_SIZE)
    dataset = ImagesDataset(ann_file, transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

    resnet = models.resnet50(pretrained=False)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool', # mettre feat5 comme OpenSelfSup ?
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    exp_name = f'r50_{IMAGE_SIZE}_bs{BATCH_SIZE}'
    exp_path = f'/home/david/byol-pytorch/examples/lightning/lightning_logs/{exp_name}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_path,
        save_top_k=5,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix='',
        save_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        '/home/david/byol-pytorch/examples/lightning/lightning_logs/',
        name=exp_name,
        version='./')

    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus = [3],
        max_epochs = EPOCHS,
        # accumulate_grad_batches = 8,
        sync_batchnorm = False,
        checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        callbacks=[lr_logger],
        # resume_from_checkpoint='/home/david/byol-pytorch/examples/lightning/lightning_logs/r50_320_bs64/last.ckpt'
    )

    trainer.fit(model, train_loader)


