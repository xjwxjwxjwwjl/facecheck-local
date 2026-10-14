# FaceCheck Local — model card

## Model

Binary face-image classifier combining two branches:

1. EfficientNet-B0, initialized with ImageNet-1K V1 weights and fine-tuned on real/StyleGAN face images using horizontal flips, JPEG re-encoding and resizing.
2. Frozen DINOv2 ViT-S/14 encoder, L2-normalized 384-dimensional CLS features and a trained logistic linear head.

The final fake probability is `0.5 * p_B0 + 0.5 * p_DINO`; classification threshold is 0.5. Inputs are EXIF-corrected RGB, resized to 224×224 with bicubic interpolation and normalized with ImageNet means/standard deviations. The CUDA DINO branch uses mixed precision; CPU uses float32, so tiny numeric differences are possible.

## Intended use

Local, exploratory analysis of face images. This is an image classifier, not an identity recognition system. The supplied interface accepts JPEG/PNG, supports batch inference and CSV export, and binds only to 127.0.0.1.

## Training and evaluation boundaries

Downstream training used 96,115 labeled images; a fixed internal validation split contained 23,885 images. B0 was fine-tuned for one epoch; DINOv2 remained frozen and only its linear classifier was trained. The blend weight was selected using internal validation perturbations. The raw data, image lists and experiment materials are not included in this inference release.

This release makes no independent benchmark, universal deepfake-detection, or unseen-generator accuracy claim. There was no independent external holdout after model selection. New diffusion models, face-swap methods, heavy compression, alignment differences, and non-face content may perform differently. A large probability is a model score, not an authenticated likelihood or forensic proof. Demographic subgroup performance and calibration were not established.

## Distribution

Download the inference-only model from the versioned GitHub Release using `download_weights.py`. Filename, byte size and SHA256 are recorded in `model_manifest.json`. The root MIT license applies to original code; third-party and data-rights context is described in THIRD_PARTY_NOTICES.md. No claim of exclusive authorship or ownership over upstream pretrained components is made.
