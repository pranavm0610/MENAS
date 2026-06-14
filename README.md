# Skin Lesion Segmentation

This repository contains the codebase for training a skin lesion segmentation model using a Multi-Encoder UNet architecture with Neural Architecture Search (NAS). The model uses a Wavelet Scattering Transform and a learnable Gabor filter bank to capture complex texture and edge features in dermatological images.

## Project Structure

```
skin-lesion/
├── dataset.py                # PyTorch Dataset class (PH2DatasetV2) with Wavelet Scattering
├── models/
│   ├── __init__.py
│   └── MultiEncDecNAS_v2.py  # The NAS-based multi-encoder U-Net model
├── train.py                  # Main training loop with NAS metrics and temperature annealing
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules
```

## Requirements

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The model is configured to use the PH2 Dataset. 
1. Download the PH2 Dataset.
2. Place it in the `PH2Dataset` directory in the project root, such that the images are located at `PH2Dataset/PH2_Dataset_images`.

## Training

To train the model, run the `train.py` script:

```bash
python train.py
```

The script will:
- Load the dataset and apply Wavelet Scattering Transform.
- Train the model using a combination of Dice loss, Focal loss, KL Divergence (for NAS), and Sparsity loss.
- Print architecture and Gabor bank parameter summaries periodically.
- Save a combined training plot to `training_results_nas_v2.png`.
- Save the final model weights to `model_nas_v2.pth`.
