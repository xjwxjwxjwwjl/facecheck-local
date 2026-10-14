"""Inference primitives for the released B0 + DINOv2 face classifier."""
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0


def build_model(pretrained=False):
    if pretrained:
        raise ValueError('This inference package only loads released local weights.')
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


def transform(size=224):
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


def load_checkpoint(path, device='cpu'):
    if not Path(path).is_file():
        raise FileNotFoundError('Model missing. Run: python download_weights.py')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('positive_class') != 'fake' or state.get('kind') != 'b0_dino_blend':
        raise ValueError('Unsupported model file or class direction.')
    from models.blend import BlendModel
    model = BlendModel(state['config']['b0_weight']).to(device)
    model.load_state_dict(state['model'])
    return model.eval(), state
