from torch.utils.data import Dataset
import torch
import json
import numpy as np
import cv2

class ImagesDataset(Dataset):
    def __init__(self, ann_file, transform):
        super().__init__()
        with open(ann_file, 'r') as f:
            anns = json.load(f)

        self.paths = [ann['file_name'] for ann in anns['images']]
        self.transform = transform

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
