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

import json
import cv2
import numpy as np

# test model, a resnet 50


# constants

BATCH_SIZE = 4
EPOCHS     = 150
LR         = 3e-4
# LR         = 1e-4
NUM_GPUS   = 1
IMAGE_SIZE = (1500, 832)
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


# main
if __name__ == '__main__':
    # dataset
    ann_file = '/home/david/AZmed-ai/az_annot_files/samples/s1_ccfrtr.json'

    # minimalist pipeline

    dataset = ImagesDataset(ann_file, IMAGE_SIZE)
    train_loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)

    resnet = models.resnet50(pretrained=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool', # mettre feat5 comme OpenSelfSup ?
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    exp_name = 'r50_1500_832'
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
        gpus = [0],
        max_epochs = EPOCHS,
        accumulate_grad_batches = 8,
        sync_batchnorm = False,
        checkpoint_callback=checkpoint_callback,
        logger=tb_logger,
        callbacks=[lr_logger]
    )

    trainer.fit(model, train_loader)
