import argparse
import os

import numpy as np
from PIL import Image

from pipeline import MENASPipeline


def main():
    parser = argparse.ArgumentParser(description='MENAS Skin Lesion Segmentation Inference')
    parser.add_argument('--image', required=True, help='Path to a dermoscopic image')
    parser.add_argument('--output_dir', default='results/', help='Directory for output files')
    parser.add_argument('--mask', default=None, help='Path to ground truth mask (optional, for evaluation)')
    parser.add_argument('--checkpoint', default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarisation threshold')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading model...')
    pipeline = MENASPipeline(checkpoint_path=args.checkpoint)
    print(f'Model loaded (trained for {pipeline.epoch} epochs)')

    image = Image.open(args.image).convert('RGB')
    image_name = os.path.splitext(os.path.basename(args.image))[0]

    print('Running inference...')
    pred_mask, _ = pipeline.predict(image, threshold=args.threshold)

    mask_path = os.path.join(args.output_dir, f'{image_name}_mask.png')
    Image.fromarray(pred_mask).save(mask_path)
    print(f'Predicted mask saved to {mask_path}')

    overlay = pipeline.create_overlay(image, pred_mask)
    overlay_path = os.path.join(args.output_dir, f'{image_name}_overlay.png')
    overlay.save(overlay_path)
    print(f'Overlay saved to {overlay_path}')

    if args.mask:
        gt = np.array(Image.open(args.mask).convert('L').resize(image.size, Image.NEAREST))
        metrics = pipeline.compute_metrics(pred_mask, gt)
        print(f'\nEvaluation metrics:')
        print(f'  Accuracy: {metrics["accuracy"]:.4f}')
        print(f'  F1:       {metrics["f1"]:.4f}')
        print(f'  IoU:      {metrics["iou"]:.4f}')
        print(f'  Dice:     {metrics["dice"]:.4f}')


if __name__ == '__main__':
    main()
