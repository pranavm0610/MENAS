import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

from dataset import PH2DatasetV2
from models.MultiEncDecNAS_v2 import MultiEncoderUNetNAS_V2


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-BCE_loss)
        return self.alpha * (1 - pt) ** self.gamma * BCE_loss


def compute_metrics(outputs, masks, threshold=0.5):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).int()
    correct = (preds == masks).int()
    accuracy = correct.sum() / correct.numel()

    preds_flat = preds.view(-1).cpu().numpy()
    masks_flat = masks.view(-1).cpu().numpy()

    f1 = f1_score(masks_flat, preds_flat, zero_division=1)

    intersection = ((preds == 1) & (masks == 1)).sum().item()
    union = ((preds == 1) | (masks == 1)).sum().item()
    iou = intersection / union if union > 0 else 1.0

    return accuracy.item(), f1, iou


def get_temperature(epoch, num_epochs, tau_start=5.0, tau_end=0.1):
    """Exponential annealing of Gumbel temperature."""
    decay = (tau_end / tau_start) ** (1.0 / max(num_epochs - 1, 1))
    return tau_start * (decay ** epoch)


def get_beta(epoch, num_epochs, warmup_fraction=0.1):
    """Linear warmup of β for KL loss."""
    warmup_epochs = max(int(num_epochs * warmup_fraction), 1)
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0


def log_architecture(model, epoch):
    """Print learned graph structure for inspection."""
    pgm = model.bottleneck.pgm
    edge_probs = torch.sigmoid(pgm.edge_prior_logits).detach().cpu().numpy()
    edge_names = ['img→edge', 'img→tex', 'edge→img', 'edge→tex', 'tex→img', 'tex→edge']

    print(f"\n{'='*60}")
    print(f"  Architecture at Epoch {epoch + 1}")
    print(f"{'='*60}")
    print("  Edge Prior Probabilities:")
    for name, prob in zip(edge_names, edge_probs):
        bar = '█' * int(prob * 20) + '░' * (20 - int(prob * 20))
        print(f"    {name:<12s}  {bar}  {prob:.3f}")
    print(f"{'='*60}\n")


def log_gabor_params(model, epoch):
    """Print Gabor filter bank parameter summary."""
    gb = model.gabor_bank
    sigmas = gb.sigmas.detach().cpu().numpy()
    thetas = gb.thetas.detach().cpu().numpy()
    lambdas = gb.lambdas.detach().cpu().numpy()
    gammas = gb.gammas.detach().cpu().numpy()

    print(f"\n{'─'*60}")
    print(f"  Gabor Bank Parameters at Epoch {epoch + 1}")
    print(f"{'─'*60}")
    print(f"  {'Filter':<8s} {'σ':>8s} {'θ (deg)':>10s} {'λ':>8s} {'γ':>8s}")
    print(f"  {'─'*44}")
    for i in range(len(sigmas)):
        theta_deg = (thetas[i] * 180.0 / 3.14159)
        print(f"  {i:<8d} {sigmas[i]:8.3f} {theta_deg:10.1f} {lambdas[i]:8.3f} {gammas[i]:8.3f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    LAMBDA_SPARSITY = 0.01     # L1 sparsity weight
    BETA_MAX = 0.1             # max KL weight after warmup
    LOG_ARCH_EVERY = 5         # log architecture every N epochs

    # V2-specific dataset parameters
    N_TEX_CHANNELS = 8         # wavelet scattering output channels
    GABOR_OUT_CHANNELS = 8     # learnable Gabor output channels
    SCATTER_J = 3              # scattering scales
    SCATTER_L = 4              # scattering orientations
    IMG_SIZE = 256

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")

    root_dir = 'PH2Dataset/PH2_Dataset_images'
    if not os.path.exists(root_dir):
        print(f"Warning: Dataset path '{root_dir}' not found. Please adjust path.")

    print("Loading dataset with Wavelet Scattering Transform...")
    print(f"  Scattering: J={SCATTER_J}, L={SCATTER_L}")
    print(f"  Texture channels: {N_TEX_CHANNELS}")
    print(f"  Gabor output channels: {GABOR_OUT_CHANNELS}")

    dataset = PH2DatasetV2(
        root_dir=root_dir,
        img_size=IMG_SIZE,
        n_tex_channels=N_TEX_CHANNELS,
        scatter_J=SCATTER_J,
        scatter_L=SCATTER_L,
    )

    val_size = int(VAL_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Quick data check
    sample_img, sample_edge, sample_tex, sample_mask = dataset[0]
    print(f"\nData shapes:")
    print(f"  Image:   {sample_img.shape}")     # (3, 256, 256)
    print(f"  Edge:    {sample_edge.shape}")     # (1, 256, 256)
    print(f"  Texture: {sample_tex.shape}")      # (8, 256, 256)
    print(f"  Mask:    {sample_mask.shape}")      # (1, 256, 256)

    model = MultiEncoderUNetNAS_V2(
        n_channels_img=3,
        n_channels_edge=1,          # raw grayscale input
        n_channels_tex=N_TEX_CHANNELS,
        n_classes=1,
        gabor_orientations=4,
        gabor_scales=3,
        gabor_kernel_sizes=[5, 9, 13],
        gabor_out_channels=GABOR_OUT_CHANNELS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gabor_params = sum(p.numel() for p in model.gabor_bank.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total:     {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Gabor bank: {gabor_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    focal_criterion = FocalLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'train_iou': [], 'val_iou': [],
        'kl_loss': [], 'sparsity_loss': [], 'temperature': [],
    }

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        temperature = get_temperature(epoch, NUM_EPOCHS)
        beta = get_beta(epoch, NUM_EPOCHS) * BETA_MAX

        history['temperature'].append(temperature)

        model.train()
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        train_iou = 0
        epoch_kl = 0
        epoch_sparsity = 0

        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}] Train (τ={temperature:.2f})')

        for images, edges, textures, masks in loop:
            images = images.to(DEVICE)
            edges = edges.to(DEVICE)
            textures = textures.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass with temperature
            outputs = model(images, edges, textures, temperature=temperature)

            probs = torch.sigmoid(outputs)

            # Task loss
            dice = dice_loss(probs, masks)
            focal = focal_criterion(probs, masks)
            task_loss = dice + focal

            # NAS losses (ELBO)
            kl_loss = model.bottleneck.get_kl_loss()
            sparsity_loss = model.bottleneck.get_sparsity_loss()

            # Total loss
            loss = task_loss + beta * kl_loss + LAMBDA_SPARSITY * sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            acc, f1, iou = compute_metrics(outputs, masks)
            train_loss += task_loss.item()
            train_acc += acc
            train_f1 += f1
            train_iou += iou
            epoch_kl += kl_loss.item()
            epoch_sparsity += sparsity_loss.item()

            loop.set_postfix(loss=loss.item(), acc=acc, f1=f1, kl=kl_loss.item())

        n_batches = len(train_loader)
        avg_train_loss = train_loss / n_batches
        avg_train_acc = train_acc / n_batches
        avg_train_f1 = train_f1 / n_batches
        avg_train_iou = train_iou / n_batches
        history['kl_loss'].append(epoch_kl / n_batches)
        history['sparsity_loss'].append(epoch_sparsity / n_batches)

        model.eval()
        val_loss = 0
        val_acc = 0
        val_f1 = 0
        val_iou = 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}] Validation')
            for images, edges, textures, masks in val_loop:
                images = images.to(DEVICE)
                edges = edges.to(DEVICE)
                textures = textures.to(DEVICE)
                masks = masks.to(DEVICE)

                # Use low temperature at eval for near-discrete selection
                outputs = model(images, edges, textures, temperature=0.1)
                probs = torch.sigmoid(outputs)

                dice = dice_loss(probs, masks)
                focal = focal_criterion(probs, masks)
                loss = dice + 0.2 * focal

                acc, f1, iou = compute_metrics(outputs, masks)
                val_loss += loss.item()
                val_acc += acc
                val_f1 += f1
                val_iou += iou

                val_loop.set_postfix(loss=loss.item(), acc=acc, f1=f1)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] '
              f'Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, F1: {avg_train_f1:.4f} | '
              f'Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, F1: {avg_val_f1:.4f} | '
              f'τ={temperature:.3f} β={beta:.4f}')

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        history['train_f1'].append(avg_train_f1)
        history['val_f1'].append(avg_val_f1)
        history['train_iou'].append(avg_train_iou)
        history['val_iou'].append(avg_val_iou)

        # Log architecture + Gabor params periodically
        if (epoch + 1) % LOG_ARCH_EVERY == 0 or epoch == 0:
            log_architecture(model, epoch)
            log_gabor_params(model, epoch)

    if DEVICE.type == "mps":
        torch.mps.synchronize()
    end_time = time.time()

    print("\nTraining complete!")
    print(f"Final Training Accuracy: {avg_train_acc:.4f}, F1 Score: {avg_train_f1:.4f}")
    print(f"Final Validation Accuracy: {avg_val_acc:.4f}, F1 Score: {avg_val_f1:.4f}")
    print(f"Total Training Time: {(end_time - start_time)/60:.2f} minutes")

    epochs_range = range(1, NUM_EPOCHS + 1)

    fig, axes = plt.subplots(3, 4, figsize=(28, 18))

    # Row 1: Standard metrics
    axes[0, 0].plot(epochs_range, history['train_loss'], label='Train Loss')
    axes[0, 0].plot(epochs_range, history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Task Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, history['train_acc'], label='Train Acc')
    axes[0, 1].plot(epochs_range, history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    axes[0, 2].plot(epochs_range, history['train_f1'], label='Train F1')
    axes[0, 2].plot(epochs_range, history['val_f1'], label='Val F1')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].legend()

    axes[0, 3].plot(epochs_range, history['train_iou'], label='Train IoU')
    axes[0, 3].plot(epochs_range, history['val_iou'], label='Val IoU')
    axes[0, 3].set_xlabel('Epoch')
    axes[0, 3].set_ylabel('IoU')
    axes[0, 3].set_title('IoU')
    axes[0, 3].legend()

    # Row 2: NAS-specific metrics
    axes[1, 0].plot(epochs_range, history['kl_loss'], label='KL Loss', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Loss (PGM)')
    axes[1, 0].legend()

    axes[1, 1].plot(epochs_range, history['sparsity_loss'], label='Sparsity Loss', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('L1 Sparsity')
    axes[1, 1].set_title('Sparsity Loss')
    axes[1, 1].legend()

    axes[1, 2].plot(epochs_range, history['temperature'], label='τ (Gumbel)', color='red')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Temperature')
    axes[1, 2].set_title('Gumbel Temperature Annealing')
    axes[1, 2].set_yscale('log')
    axes[1, 2].legend()

    # Final adjacency heatmap
    pgm = model.bottleneck.pgm
    edge_probs = torch.sigmoid(pgm.edge_prior_logits).detach().cpu().numpy()
    node_names = ['img', 'edge', 'tex']
    adj_matrix = np.zeros((3, 3))
    for idx, (i, j) in enumerate(pgm.EDGE_LIST):
        adj_matrix[i, j] = edge_probs[idx]

    im = axes[1, 3].imshow(adj_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    axes[1, 3].set_xticks(range(3))
    axes[1, 3].set_yticks(range(3))
    axes[1, 3].set_xticklabels(node_names)
    axes[1, 3].set_yticklabels(node_names)
    axes[1, 3].set_title('Final Learned Adjacency (Prior)')
    axes[1, 3].set_xlabel('Target')
    axes[1, 3].set_ylabel('Source')
    for ii in range(3):
        for jj in range(3):
            axes[1, 3].text(jj, ii, f'{adj_matrix[ii, jj]:.2f}',
                           ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1, 3])

    # Row 3: Gabor filter parameter evolution (final state)
    gb = model.gabor_bank
    sigmas = gb.sigmas.detach().cpu().numpy()
    thetas_deg = gb.thetas.detach().cpu().numpy() * 180.0 / np.pi
    lambdas_val = gb.lambdas.detach().cpu().numpy()
    gammas_val = gb.gammas.detach().cpu().numpy()

    filter_indices = np.arange(len(sigmas))

    axes[2, 0].bar(filter_indices, sigmas, color='steelblue')
    axes[2, 0].set_xlabel('Filter Index')
    axes[2, 0].set_ylabel('σ (sigma)')
    axes[2, 0].set_title('Learned Gabor σ')
    axes[2, 0].set_xticks(filter_indices)

    axes[2, 1].bar(filter_indices, thetas_deg, color='coral')
    axes[2, 1].set_xlabel('Filter Index')
    axes[2, 1].set_ylabel('θ (degrees)')
    axes[2, 1].set_title('Learned Gabor θ')
    axes[2, 1].set_xticks(filter_indices)

    axes[2, 2].bar(filter_indices, lambdas_val, color='seagreen')
    axes[2, 2].set_xlabel('Filter Index')
    axes[2, 2].set_ylabel('λ (lambda)')
    axes[2, 2].set_title('Learned Gabor λ')
    axes[2, 2].set_xticks(filter_indices)

    axes[2, 3].bar(filter_indices, gammas_val, color='goldenrod')
    axes[2, 3].set_xlabel('Filter Index')
    axes[2, 3].set_ylabel('γ (gamma)')
    axes[2, 3].set_title('Learned Gabor γ')
    axes[2, 3].set_xticks(filter_indices)

    plt.tight_layout()
    plt.savefig('training_results_nas_v2.png', dpi=150)
    print("Training plot saved to training_results_nas_v2.png")

    # Save model
    torch.save(model.state_dict(), 'model_nas_v2.pth')
    print("Model saved to model_nas_v2.pth")
