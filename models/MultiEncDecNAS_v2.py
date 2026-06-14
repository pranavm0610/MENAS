import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, skips):
        x1 = self.up(x1)
        combined_skips = torch.cat(skips, dim=1)
        x = torch.cat([combined_skips, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LearnableGaborBank(nn.Module):
    def __init__(
        self,
        n_orientations=4,
        n_scales=3,
        kernel_sizes=None,
        out_channels=8,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 9, 13]
        assert len(kernel_sizes) == n_scales

        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_orientations * n_scales
        self.out_channels = out_channels
        self.max_ks = max(kernel_sizes)

        sigmas = []
        thetas = []
        lambdas = []
        gammas = []

        for s_idx, ks in enumerate(kernel_sizes):
            for o_idx in range(n_orientations):
                sigma_init = 0.56 * ks / (2.0 * math.pi)
                theta_init = o_idx * math.pi / n_orientations
                lambda_init = float(ks) / 2.0
                gamma_init = 0.5

                sigmas.append(sigma_init)
                thetas.append(theta_init)
                lambdas.append(lambda_init)
                gammas.append(gamma_init)

        self.sigmas = nn.Parameter(torch.tensor(sigmas, dtype=torch.float32))
        self.thetas = nn.Parameter(torch.tensor(thetas, dtype=torch.float32))
        self.lambdas = nn.Parameter(torch.tensor(lambdas, dtype=torch.float32))
        self.gammas = nn.Parameter(torch.tensor(gammas, dtype=torch.float32))

        self.register_buffer(
            'filter_ks',
            torch.tensor([ks for ks in kernel_sizes for _ in range(n_orientations)])
        )

        self.proj = nn.Sequential(
            nn.Conv2d(self.n_filters, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_gabor_kernel(self, sigma, theta, lambd, gamma, ks):
        half = ks // 2
        y, x = torch.meshgrid(
            torch.arange(-half, half + 1, dtype=torch.float32, device=sigma.device),
            torch.arange(-half, half + 1, dtype=torch.float32, device=sigma.device),
            indexing='ij'
        )

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t

        gaussian = torch.exp(
            -0.5 * (x_rot ** 2 + gamma ** 2 * y_rot ** 2) / (sigma ** 2 + 1e-6)
        )
        sinusoid = torch.cos(2.0 * math.pi * x_rot / (lambd + 1e-6))

        kernel = gaussian * sinusoid

        kernel = kernel - kernel.mean()

        return kernel

    def forward(self, gray_input):
        B, _, H, W = gray_input.shape
        device = gray_input.device

        edge_maps = []
        for i in range(self.n_filters):
            ks = int(self.filter_ks[i].item())
            kernel = self._make_gabor_kernel(
                self.sigmas[i], self.thetas[i],
                self.lambdas[i], self.gammas[i], ks
            )
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            pad = ks // 2
            response = F.conv2d(gray_input, kernel, padding=pad)
            edge_maps.append(response)

        edge_stack = torch.cat(edge_maps, dim=1)

        edge_stack = torch.abs(edge_stack)

        return self.proj(edge_stack)


class IdentityOp(nn.Module):
    def forward(self, x):
        return x


class Conv1x1Op(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class Conv3x3Op(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class SEOp(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


OPS_REGISTRY = [IdentityOp, Conv1x1Op, Conv3x3Op, SEOp]
NUM_OPS = len(OPS_REGISTRY)


class MixedOp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList()
        for OpClass in OPS_REGISTRY:
            if OpClass is IdentityOp:
                self.ops.append(OpClass())
            else:
                self.ops.append(OpClass(channels))

    def forward(self, x, op_weights):
        out = 0
        for w, op in zip(op_weights, self.ops):
            out = out + w * op(x)
        return out


class PGMGraphStructure(nn.Module):
    N_NODES = 3
    N_EDGES = 6
    EDGE_LIST = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        prior_init = torch.zeros(self.N_EDGES)
        for idx, (i, j) in enumerate(self.EDGE_LIST):
            if i == 0:
                prior_init[idx] = 1.5
            else:
                prior_init[idx] = 0.0
        self.edge_prior_logits = nn.Parameter(prior_init)

        self.register_buffer(
            'op_prior_logprobs',
            torch.full((NUM_OPS,), -math.log(NUM_OPS))
        )

        feat_dim = channels * self.N_NODES
        hidden = 256

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.edge_posterior = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.N_EDGES),
        )

        self.op_posterior = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.N_EDGES * NUM_OPS),
        )

        self.hierarchy_gate = nn.Sequential(
            nn.Linear(feat_dim, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, self.N_NODES),
            nn.Sigmoid(),
        )

    def _gumbel_sigmoid(self, logits, temperature, hard=False):
        u = torch.rand_like(logits).clamp(1e-8, 1.0 - 1e-8)
        gumbel_noise = -torch.log(-torch.log(u))
        y = torch.sigmoid((logits + gumbel_noise) / temperature)
        if hard:
            y_hard = (y > 0.5).float()
            y = y_hard - y.detach() + y
        return y

    def forward(self, modality_features, temperature=1.0):
        B = modality_features[0].shape[0]

        pooled = []
        for feat in modality_features:
            pooled.append(self.global_pool(feat).view(B, -1))
        global_feat = torch.cat(pooled, dim=1)

        edge_logits = self.edge_posterior(global_feat)
        edge_weights = self._gumbel_sigmoid(edge_logits, temperature)

        op_logits = self.op_posterior(global_feat)
        op_logits = op_logits.view(B, self.N_EDGES, NUM_OPS)
        op_weights = F.gumbel_softmax(op_logits, tau=temperature, hard=False, dim=-1)

        h_gates = self.hierarchy_gate(global_feat)

        kl_loss = self._compute_kl(edge_logits, op_logits)

        return edge_weights, op_weights, h_gates, kl_loss

    def _compute_kl(self, edge_logits, op_logits):
        q_edge = torch.sigmoid(edge_logits)
        p_edge = torch.sigmoid(self.edge_prior_logits)

        eps = 1e-8
        kl_edge = (
            q_edge * (torch.log(q_edge + eps) - torch.log(p_edge + eps))
            + (1 - q_edge) * (torch.log(1 - q_edge + eps) - torch.log(1 - p_edge + eps))
        )
        kl_edge = kl_edge.sum(dim=-1).mean()

        q_op = F.softmax(op_logits, dim=-1)
        log_q = torch.log(q_op + eps)
        log_p = self.op_prior_logprobs.unsqueeze(0).unsqueeze(0)
        kl_op = (q_op * (log_q - log_p)).sum(dim=-1).sum(dim=-1).mean()

        return kl_edge + kl_op


class ModalityGraphBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.n_nodes = PGMGraphStructure.N_NODES
        self.n_edges = PGMGraphStructure.N_EDGES
        self.edge_list = PGMGraphStructure.EDGE_LIST

        self.norms = nn.ModuleList([
            nn.GroupNorm(1, channels)
            for _ in range(self.n_nodes)
        ])

        self.edge_ops = nn.ModuleList([
            MixedOp(channels) for _ in range(self.n_edges)
        ])

        self.agg_attn = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 2),
            )
            for _ in range(self.n_nodes)
        ])

        self.pgm = PGMGraphStructure(channels)

        self._kl_loss = torch.tensor(0.0)

        self._incoming = {}
        for j in range(self.n_nodes):
            self._incoming[j] = [
                idx for idx, (src, dst) in enumerate(self.edge_list) if dst == j
            ]

    def get_kl_loss(self):
        return self._kl_loss

    def get_sparsity_loss(self):
        return torch.abs(torch.sigmoid(self.pgm.edge_prior_logits)).sum()

    def forward(self, features, temperature=1.0):
        B, C, H, W = features[0].shape

        normed = [self.norms[i](features[i]) for i in range(self.n_nodes)]

        edge_weights, op_weights, h_gates, kl_loss = self.pgm(normed, temperature)
        self._kl_loss = kl_loss

        messages = []
        for edge_idx, (src, dst) in enumerate(self.edge_list):
            msg = self._compute_edge_message(
                normed[src], self.edge_ops[edge_idx],
                edge_weights[:, edge_idx],
                op_weights[:, edge_idx, :],
            )
            messages.append(msg)

        refined = []
        for j in range(self.n_nodes):
            incoming_idxs = self._incoming[j]
            incoming_msgs = [messages[idx] for idx in incoming_idxs]

            stacked = torch.stack(incoming_msgs, dim=1)

            attn_input = stacked.mean(dim=1)
            attn_logits = self.agg_attn[j](attn_input)
            attn_w = F.softmax(attn_logits, dim=-1)
            attn_w = attn_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            aggregated = (stacked * attn_w).sum(dim=1)

            gate = h_gates[:, j].view(B, 1, 1, 1)
            x_refined = features[j] + gate * aggregated

            refined.append(x_refined)

        return torch.cat(refined, dim=1)

    def _compute_edge_message(self, x_src, mixed_op, edge_w, op_w):
        B = x_src.shape[0]

        op_outputs = []
        for op in mixed_op.ops:
            op_outputs.append(op(x_src))

        op_stack = torch.stack(op_outputs, dim=1)

        op_w_expanded = op_w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mixed = (op_stack * op_w_expanded).sum(dim=1)

        edge_w_expanded = edge_w.view(B, 1, 1, 1)
        return mixed * edge_w_expanded


class MultiEncoderUNetNAS_V2(nn.Module):
    def __init__(
        self,
        n_channels_img=3,
        n_channels_edge=1,
        n_channels_tex=8,
        n_classes=1,
        bilinear=True,
        gabor_orientations=4,
        gabor_scales=3,
        gabor_kernel_sizes=None,
        gabor_out_channels=8,
    ):
        super(MultiEncoderUNetNAS_V2, self).__init__()
        self.bilinear = bilinear

        self.gabor_bank = LearnableGaborBank(
            n_orientations=gabor_orientations,
            n_scales=gabor_scales,
            kernel_sizes=gabor_kernel_sizes,
            out_channels=gabor_out_channels,
        )
        edge_enc_channels = gabor_out_channels  # encoder sees Gabor output

        self.inc_img = DoubleConv(n_channels_img, 64)
        self.down1_img = Down(64, 128)
        self.down2_img = Down(128, 256)
        self.down3_img = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4_img = Down(512, 1024 // factor)

        self.inc_edge = DoubleConv(edge_enc_channels, 64)
        self.down1_edge = Down(64, 128)
        self.down2_edge = Down(128, 256)
        self.down3_edge = Down(256, 512)
        self.down4_edge = Down(512, 1024 // factor)

        self.inc_tex = DoubleConv(n_channels_tex, 64)
        self.down1_tex = Down(64, 128)
        self.down2_tex = Down(128, 256)
        self.down3_tex = Down(256, 512)
        self.down4_tex = Down(512, 1024 // factor)

        base_ch = 1024 // factor
        self.bottleneck = ModalityGraphBottleneck(base_ch)

        self.up1 = Up(3 * base_ch, 3 * 512, 512 // factor, bilinear)
        self.up2 = Up(512 // factor, 3 * 256, 256 // factor, bilinear)
        self.up3 = Up(256 // factor, 3 * 128, 128 // factor, bilinear)
        self.up4 = Up(128 // factor, 3 * 64, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, img, edge_gray, tex, temperature=1.0):
        """
        Args:
            img:       (B, 3, H, W) RGB input
            edge_gray: (B, 1, H, W) grayscale input for Gabor bank
            tex:       (B, n_tex_channels, H, W) wavelet scattering texture
            temperature: Gumbel temperature

        Returns:
            logits: (B, n_classes, H, W) segmentation logits
        """
        edge = self.gabor_bank(edge_gray)  # (B, gabor_out_channels, H, W)

        x1_img = self.inc_img(img)
        x2_img = self.down1_img(x1_img)
        x3_img = self.down2_img(x2_img)
        x4_img = self.down3_img(x3_img)
        x5_img = self.down4_img(x4_img)

        x1_edge = self.inc_edge(edge)
        x2_edge = self.down1_edge(x1_edge)
        x3_edge = self.down2_edge(x2_edge)
        x4_edge = self.down3_edge(x3_edge)
        x5_edge = self.down4_edge(x4_edge)

        x1_tex = self.inc_tex(tex)
        x2_tex = self.down1_tex(x1_tex)
        x3_tex = self.down2_tex(x2_tex)
        x4_tex = self.down3_tex(x3_tex)
        x5_tex = self.down4_tex(x4_tex)

        x = self.bottleneck([x5_img, x5_edge, x5_tex], temperature=temperature)

        x = self.up1(x, [x4_img, x4_edge, x4_tex])
        x = self.up2(x, [x3_img, x3_edge, x3_tex])
        x = self.up3(x, [x2_img, x2_edge, x2_tex])
        x = self.up4(x, [x1_img, x1_edge, x1_tex])

        logits = self.outc(x)
        return logits
