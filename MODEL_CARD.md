# FaceCheck Local — model card

## Model

Binary face-image classifier combining two branches:

1. EfficientNet-B0, initialized with ImageNet-1K V1 weights and fine-tuned on real/StyleGAN face images using horizontal flips, JPEG re-encoding and resizing.
2. Frozen DINOv2 ViT-S/14 encoder, L2-normalized 384-dimensional CLS features and a trained logistic linear head.

The final fake probability is `0.5 * p_B0 + 0.5 * p_DINO`; classification threshold is 0.5. Inputs are EXIF-corrected RGB, resized to 224×224 with bicubic interpolation and normalized with ImageNet means/standard deviations. The CUDA DINO branch uses mixed precision; CPU uses float32, so tiny numeric differences are possible.

## Intended use

Local, exploratory analysis of face images. This is an image classifier, not an identity recognition system. The supplied interface accepts JPEG/PNG, supports batch inference and CSV export, and binds only to 127.0.0.1.

## Training and evaluation boundaries

Downstream training used 96,115 labeled images, with 23,885 images in a fixed group-aware internal validation split. In v0.2.0, B0 received two additional epochs after its original one-epoch training. The added learning rates were 3e-5 and 1.5e-5, with AdamW, batch size 32 and mixed precision. Augmentations included horizontal flips, down/up-sampling to 64–192 pixels, JPEG quality 40–95 and occasional light Gaussian blur.

DINOv2 remained frozen. Its logistic linear classifier (C=1) was fitted on clean training features and one augmented feature view per training image, each weighted 0.5. The fixed probability blend and threshold are both 0.5.

Six candidates were compared on five perturbation conditions using 4,096 internal images. The selected model was then evaluated on the full internal split and four additional perturbations. Internal validation had already been used in earlier experiments. These comparisons therefore retain model-selection bias. This refinement used one training seed.

| Internal condition | v0.1.0 accuracy | v0.2.0 accuracy |
|---|---:|---:|
| Full clean, 23,885 images | 99.7865% | 99.8577% |
| JPEG Q50, 4,096 development images | 96.2402% | 98.9990% |
| Resize64, same development images | 89.1113% | 97.2656% |

The full internal ROC-AUC was 0.999947413. Compression and resizing measurements describe artificial perturbation robustness on the original domain. They do not establish performance on unseen generators.

This release makes no independent benchmark, universal deepfake-detection, or unseen-generator accuracy claim. There was no independent external holdout after model selection. New diffusion models, face-swap methods, heavy compression, alignment differences, and non-face content may perform differently. A large probability is a model score, not an authenticated likelihood or forensic proof. Demographic subgroup performance and calibration were not established.

## Distribution

Download the inference-only model from the versioned GitHub Release using `download_weights.py`. Filename, byte size and SHA256 are recorded in `model_manifest.json`. The root MIT license applies to original code; third-party and data-rights context is described in THIRD_PARTY_NOTICES.md. No claim of exclusive authorship or ownership over upstream pretrained components is made.
