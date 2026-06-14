

import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class WaveletScatteringTransform:
    """
    Two-layer wavelet scattering transform using Morlet wavelets.

    Layer 0: |x ★ φ_J|                          → low-pass (1 channel)
    Layer 1: |x ★ ψ_{j,l}| ★ φ_J               → J×L channels
    Layer 2: ||x ★ ψ_{j1,l1}| ★ ψ_{j2,l2}| ★ φ_J  → second-order
             (only j2 > j1 for non-redundancy)

    All convolutions use FFT for efficiency.

    Parameters
    ----------
    J : int
        Number of dyadic scales (default 3 → scales 2^1, 2^2, 2^3).
    L : int
        Number of orientations (default 4 → 0, π/4, π/2, 3π/4).
    img_size : int
        Spatial size of input images (assumed square).
    out_channels : int
        Number of output channels after 1×1 projection (default 8).
    """

    def __init__(self, J=3, L=4, img_size=256, out_channels=8):
        self.J = J
        self.L = L
        self.img_size = img_size
        self.out_channels = out_channels

        # Pre-compute filter bank in the frequency domain
        self._filters = self._build_filters(J, L, img_size)

        # Compute total number of scattering coefficients
        self.n_order0 = 1  # low-pass
        self.n_order1 = J * L
        self.n_order2 = 0
        for j1 in range(J):
            for j2 in range(j1 + 1, J):
                self.n_order2 += L * L
        self.n_coeffs = self.n_order0 + self.n_order1 + self.n_order2

        # 1×1 projection to desired output channels (static, no grad)
        # We'll just use a simple linear mix at call time via numpy
        np.random.seed(42)
        self._proj = np.random.randn(out_channels, self.n_coeffs).astype(np.float32)
        # Normalise rows for stability
        norms = np.linalg.norm(self._proj, axis=1, keepdims=True) + 1e-8
        self._proj = self._proj / norms

    def _morlet_wavelet_2d(self, N, sigma, theta, xi, slant=0.5):
        """Create a 2D Morlet wavelet in the spatial domain, return its FFT."""
        grid_y, grid_x = np.mgrid[-N//2:N//2, -N//2:N//2].astype(np.float64)

        # Rotation
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = cos_t * grid_x + sin_t * grid_y
        y_rot = -sin_t * grid_x + cos_t * grid_y

        # Gaussian envelope
        gaussian = np.exp(
            -0.5 * (x_rot**2 / sigma**2 + y_rot**2 / (slant * sigma)**2)
        )

        # Oscillation
        oscillation = np.exp(1j * xi * x_rot)

        # Subtract DC to make it admissible (zero mean)
        dc_correction = np.exp(-0.5 * xi**2 * sigma**2)

        wavelet = gaussian * (oscillation - dc_correction)

        # FFT shift and transform to frequency domain
        wavelet_fft = np.fft.fft2(np.fft.ifftshift(wavelet))
        return wavelet_fft

    def _gaussian_filter_2d(self, N, sigma):
        """Low-pass Gaussian filter in frequency domain."""
        grid_y, grid_x = np.mgrid[-N//2:N//2, -N//2:N//2].astype(np.float64)
        gaussian = np.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma**2)
        gaussian = gaussian / gaussian.sum()
        gaussian_fft = np.fft.fft2(np.fft.ifftshift(gaussian))
        return gaussian_fft

    def _build_filters(self, J, L, N):
        """Build the Morlet wavelet filter bank + low-pass filter."""
        filters = {}

        # Low-pass filter at scale J
        sigma_low = 0.8 * (2**J)
        filters['phi'] = self._gaussian_filter_2d(N, sigma_low)

        # Bandpass wavelets
        xi_0 = 3.0 / 4.0 * math.pi  # mother wavelet frequency
        sigma_0 = 0.8

        filters['psi'] = []
        for j in range(J):
            for l in range(L):
                sigma_j = sigma_0 * (2**j)
                xi_j = xi_0 / (2**j)
                theta_l = l * math.pi / L

                psi_fft = self._morlet_wavelet_2d(N, sigma_j, theta_l, xi_j)
                filters['psi'].append({
                    'j': j, 'l': l, 'fft': psi_fft
                })

        return filters

    def _apply_filter_fft(self, x_fft, filter_fft):
        """Apply a filter in the frequency domain and return spatial result."""
        result_fft = x_fft * filter_fft
        result = np.real(np.fft.ifft2(result_fft))
        return result

    def __call__(self, image):
        """
        Compute the wavelet scattering transform of a grayscale image.

        Parameters
        ----------
        image : PIL.Image
            Input image (will be converted to grayscale).

        Returns
        -------
        torch.Tensor
            Scattering coefficients of shape (out_channels, H, W).
        """
        # Convert to grayscale numpy
        img_np = np.array(image.convert("L")).astype(np.float64)

        # Resize to expected size
        if img_np.shape[0] != self.img_size or img_np.shape[1] != self.img_size:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img_np.astype(np.uint8))
            img_pil = img_pil.resize((self.img_size, self.img_size))
            img_np = np.array(img_pil).astype(np.float64)

        # Normalise to [0, 1]
        img_np = img_np / 255.0

        N = self.img_size
        phi_fft = self._filters['phi']
        psi_list = self._filters['psi']

        # FFT of input
        x_fft = np.fft.fft2(img_np)

        S0 = self._apply_filter_fft(x_fft, phi_fft)
        coeffs = [S0]

        U1_list = []
        for psi in psi_list:
            U1 = np.abs(self._apply_filter_fft(x_fft, psi['fft']))
            U1_list.append((psi['j'], psi['l'], U1))

            # Low-pass the modulus
            U1_fft = np.fft.fft2(U1)
            S1 = self._apply_filter_fft(U1_fft, phi_fft)
            coeffs.append(S1)

        for j1, l1, U1 in U1_list:
            U1_fft = np.fft.fft2(U1)
            for psi2 in psi_list:
                j2 = psi2['j']
                if j2 <= j1:
                    continue  # non-redundant: j2 > j1
                U2 = np.abs(self._apply_filter_fft(U1_fft, psi2['fft']))
                U2_fft = np.fft.fft2(U2)
                S2 = self._apply_filter_fft(U2_fft, phi_fft)
                coeffs.append(S2)

        # Stack all coefficients: (n_coeffs, N, N)
        coeffs = np.stack(coeffs, axis=0).astype(np.float32)

        # Project to out_channels via fixed random projection
        # coeffs: (n_coeffs, N, N) → reshaped for matmul
        C, H, W = coeffs.shape
        flat = coeffs.reshape(C, -1)             # (n_coeffs, H*W)
        projected = self._proj @ flat             # (out_channels, H*W)
        projected = projected.reshape(self.out_channels, H, W)

        # Normalise each channel to [0, 1]
        for c in range(self.out_channels):
            ch = projected[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                projected[c] = (ch - ch_min) / (ch_max - ch_min)
            else:
                projected[c] = 0.0

        return torch.from_numpy(projected)  # (out_channels, H, W)


class PH2DatasetV2(Dataset):
    """
    PH2 dataset with advanced edge and texture modalities.

    Returns
    -------
    image : Tensor (3, 256, 256)
        RGB image, ImageNet-normalised.
    edge_gray : Tensor (1, 256, 256)
        Grayscale image in [0, 1] — the Learnable Gabor Bank inside the
        model will extract multi-scale edges from this.
    texture : Tensor (n_tex_channels, 256, 256)
        Wavelet scattering texture coefficients.
    mask : Tensor (1, 256, 256)
        Binary segmentation mask.
    """

    def __init__(
        self,
        root_dir,
        img_size=256,
        n_tex_channels=8,
        scatter_J=3,
        scatter_L=4,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.n_tex_channels = n_tex_channels

        # Image transform (same as original)
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Grayscale transform for edge input
        self.gray_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0, 1]
        ])

        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # Wavelet scattering for texture
        self.scattering = WaveletScatteringTransform(
            J=scatter_J,
            L=scatter_L,
            img_size=img_size,
            out_channels=n_tex_channels,
        )

        # Patient folders
        self.patient_folders = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        patient_folder = self.patient_folders[idx]
        dermo_path = os.path.join(
            self.root_dir, patient_folder,
            f"{patient_folder}_Dermoscopic_Image"
        )
        seg_path = os.path.join(
            self.root_dir, patient_folder,
            f"{patient_folder}_lesion"
        )

        image_name = os.listdir(dermo_path)[0]
        mask_name = os.listdir(seg_path)[0]

        image = Image.open(os.path.join(dermo_path, image_name)).convert('RGB')
        mask = Image.open(os.path.join(seg_path, mask_name)).convert('L')

        texture = self.scattering(image)  # (n_tex_channels, H, W)

        edge_gray = self.gray_transform(image)  # (1, H, W)

        image_tensor = self.image_transform(image)  # (3, H, W)

        mask_tensor = self.mask_transform(mask)  # (1, H, W)
        mask_tensor = (mask_tensor > 0).float()

        return image_tensor, edge_gray, texture, mask_tensor
