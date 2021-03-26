from mmcv import Config
from mmdet.models import build_detector
import torch
from byol_pytorch import BYOLFPN

cfg = Config.fromfile('/home/david/AZmed-ai/az_configs/trauma/retinanet_r50.py')

model = build_detector(cfg.model)

class Stacker(torch.nn.Module):
    def forward(self, x):
        return torch.stack([_x.mean([2, 3]) for _x in x]).transpose(0, 1)

resnet = model.backbone
fpn = model.neck

extractor = torch.nn.Sequential(
    resnet,
    fpn,
    Stacker()
)

extractor.to('cuda:3')

learner = BYOLFPN(
    extractor,
    image_size = 1280,
    hidden_layer = -1,
    projection_hidden_size=2048
).to('cuda:3')