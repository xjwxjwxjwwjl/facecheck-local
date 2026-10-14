"""Probability average of robust B0 and frozen official DINOv2-S/14 head."""
import sys
from pathlib import Path
import torch
from torch import nn


class BlendModel(nn.Module):
    def __init__(self,b0_weight=.5):
        super().__init__()
        from core import build_model
        base=Path(__file__).resolve().parents[1]
        candidates=[base/'vendor/dinov2']
        repo=next((p for p in candidates if (p/'dinov2/hub/backbones.py').exists()),None)
        if repo is None:raise FileNotFoundError('Pinned official DINOv2 source missing; see model package vendor/')
        sys.path.insert(0,str(repo))
        from dinov2.hub.backbones import dinov2_vits14
        self.b0=build_model(False)
        self.dino=dinov2_vits14(pretrained=False)
        self.head=nn.Linear(384,1)
        if not 0<=b0_weight<=1:raise ValueError('Invalid mixture weight')
        self.b0_weight=float(b0_weight)

    def forward(self,x):
        p_b0=self.b0(x).flatten().sigmoid()
        # Matches the frozen-feature extraction precision on CUDA; CPU uses float32.
        with torch.autocast(device_type=x.device.type,enabled=x.device.type=='cuda'):
            feat=self.dino(x)
        feat=torch.nn.functional.normalize(feat.float(),dim=1)
        p_dino=self.head(feat).flatten().sigmoid()
        p=self.b0_weight*p_b0+(1-self.b0_weight)*p_dino
        return torch.logit(p.clamp(1e-7,1-1e-7)).unsqueeze(1)
