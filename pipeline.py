import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from models.MultiEncDecNAS_v2 import MultiEncoderUNetNAS_V2
from dataset import WaveletScatteringTransform


class MENASPipeline:
    """Wraps model loading and prediction for the Multi-Encoder NAS U-Net."""

    def __init__(self, checkpoint_path='models/best_model.pth', device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_cfg = checkpoint['model_config']
        scatter_cfg = checkpoint['scatter_config']
        self.img_size = scatter_cfg['img_size']

        self.model = MultiEncoderUNetNAS_V2(**model_cfg).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.scattering = WaveletScatteringTransform(
            J=scatter_cfg['scatter_J'],
            L=scatter_cfg['scatter_L'],
            img_size=self.img_size,
            out_channels=scatter_cfg['n_tex_channels'],
        )

        self.image_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.gray_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

        self.metrics = checkpoint.get('metrics', {})
        self.epoch = checkpoint.get('epoch', 0)

    def preprocess(self, image):
        img_tensor = self.image_transform(image).unsqueeze(0)
        edge_tensor = self.gray_transform(image).unsqueeze(0)
        tex_tensor = self.scattering(image).unsqueeze(0)
        return img_tensor, edge_tensor, tex_tensor

    @torch.no_grad()
    def predict(self, image, threshold=0.5):
        """
        Run segmentation on a PIL Image.

        Returns:
            mask: numpy array (H, W) uint8, 0 or 255, at original resolution
            prob_map: numpy array (H, W) float, at model resolution
        """
        orig_size = image.size  # (W, H)

        img_t, edge_t, tex_t = self.preprocess(image)
        img_t = img_t.to(self.device)
        edge_t = edge_t.to(self.device)
        tex_t = tex_t.to(self.device)

        logits = self.model(img_t, edge_t, tex_t, temperature=0.1)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        mask = (prob_map > threshold).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask).resize(orig_size, Image.NEAREST)

        return np.array(mask_pil), prob_map

    @staticmethod
    def compute_metrics(pred_mask, gt_mask):
        pred = (pred_mask > 127).astype(np.float32).flatten()
        gt = (gt_mask > 127).astype(np.float32).flatten()

        intersection = (pred * gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        union = pred_sum + gt_sum - intersection

        dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
        iou = intersection / union if union > 0 else 1.0
        accuracy = (pred == gt).sum() / len(pred)

        return {'dice': float(dice), 'iou': float(iou),
                'accuracy': float(accuracy), 'f1': float(dice)}

    @staticmethod
    def create_overlay(image, mask, alpha=0.4, color=(0, 255, 0)):
        image_np = np.array(image.convert('RGB'))
        mask_bool = mask > 127
        overlay = image_np.copy()
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
        ).astype(np.uint8)
        return Image.fromarray(overlay)
