# Third-party components

The root MIT license covers original application and inference glue code only.
It does not replace the terms of third-party code, pretrained weights, or datasets.

- **DINOv2 ViT-S/14** — Meta, Apache-2.0. A minimal, unmodified backbone source subset is vendored under `vendor/dinov2`, pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Its LICENSE and original MODEL_CARD are preserved. Repository: https://github.com/facebookresearch/dinov2 . Original encoder weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth . Pretraining corpus: LVD-142M.
- **EfficientNet-B0 / torchvision** — torchvision, BSD-3-Clause; initialized from torchvision `IMAGENET1K_V1` pretrained weights before downstream training. Documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html . ImageNet dataset terms are separate from the torchvision code license.
- **PyTorch** — BSD-style terms and bundled third-party notices. Upstream license copies are in `licenses/`.

No training images, face thumbnails, dataset archives, or evaluation image lists are distributed here. The downstream model was trained using a supplied real-vs-fake face dataset (real/StyleGAN labels); the complete image-rights chain was not independently audited. Publication of the model does not grant rights to the underlying training images, and this repository does not claim that the model is cleared for every commercial or biometric use.

The inference-only checkpoint contains parameters and minimal architecture/class metadata, with optimizer, training paths, experiment identifiers and validation records removed. See MODEL_CARD.md for capabilities and limitations.
