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

class MultiViewDataset(Dataset):
    
    """
    Args:
        df: DataFrame where index is a patient unique id 
        path_col: column of image paths
    """
    
    def __init__(self, df, path_col, image_size, only_multiview=False, only_single=False):
        super().__init__()
        
        self.pairs = []
        count_pairs = 0
        count_single = 0
        
        for idx, df_patient in df.groupby(level=0):
            images = df_patient[path_col].unique()
            patient_pairs = list(itertools.combinations(images, 2))            
            if patient_pairs and not only_single:
                self.pairs += patient_pairs
                count_pairs += len(patient_pairs) * 2
            elif not only_multiview:
                self.pairs.append((images[0], ))
                count_single += 1
                
        print(f'{count_pairs} pairs of images')
        print(f'{count_single} single images')
        

        self.transform_hard = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.8, 1.)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=(-20, 20), expand=False),
            NormalizeMeanVar()
        ])
        
        self.transform_light = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.9, 1.)),
            NormalizeMeanVar()
        ])

    def __len__(self):
        return len(self.pairs)

    def read_and_convert(self, path):
        img = cv2.imread(path, -1).astype(np.float32)
        img = torch.tensor(img).unsqueeze(0)
        img = img.expand(3, -1, -1).unsqueeze(0)
        return img

    def __getitem__(self, index):
        
        pair = self.pairs[index]
                
        if len(pair) > 1:
            path1, path2 = pair

            img1 = self.read_and_convert(path1)
            img2 = self.read_and_convert(path2)

            view1 = self.transform_light(img1)
            view2 = self.transform_light(img2)        
        else:
            path = pair[0]
            img = self.read_and_convert(path)
            
            view1 = self.transform_hard(img)
            view2 = self.transform_hard(img)  
            
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

    dataset = MultiViewDataset(df, '0', 1280)
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
    ).to('cuda:3')

    model = SelfSupervisedLearner(
        fpn_extractor,
        image_size = IMAGE_SIZE,
        hidden_layer = -1,
        projection_size = 256,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd() + '/fpn_multiview',
        save_top_k=10,
        verbose=True,
        monitor='loss',
        mode='min',
        prefix=''
    )

    trainer = pl.Trainer(
        gpus = [3],
        max_epochs = EPOCHS,
        accumulate_grad_batches = 8,
        sync_batchnorm = False,
        checkpoint_callback=checkpoint_callback
    )

    trainer.fit(model, train_loader)

