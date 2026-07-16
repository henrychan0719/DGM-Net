"""DGM-Net (Directional Geometric Mamba Network) core architecture.

Paper: "Breaking the Resource Wall: Geometry-Guided Sequence Modeling
for Efficient Semantic Segmentation" (arXiv:2604.23399).

ResNet-101 backbone (output stride 8) + DGM-Module geometric priors
+ Cascade GN-SSM (2x spatial Mamba + 1x G-Mamba)
+ Geometry-aware Offset-Aligned Decoding (GOAD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mamba_ssm import Mamba


class SpatialMambaBlock(nn.Module):
    """Pure spatial scanning layer (L1 / L2).

    Runs four directional Mamba scans (LR / RL / TD / DT) without DGM
    intervention, focusing on building macro context and semantic layout.
    """

    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super(SpatialMambaBlock, self).__init__()
        self.mamba_lr = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_rl = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_td = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_dt = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Gated fusion of the four directional scans
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model * 4),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj_out = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        def get_seq(feat, flip=False, trans=False):
            if trans: feat = feat.transpose(2, 3).contiguous()
            s = feat.reshape(B, C, N).permute(0, 2, 1)
            return torch.flip(s, [1]) if flip else s

        out_lr = self.mamba_lr(get_seq(x)).permute(0, 2, 1).view(B, C, H, W)
        out_rl = torch.flip(self.mamba_rl(get_seq(x, True)), [1]).permute(0, 2, 1).view(B, C, H, W)
        out_td = self.mamba_td(get_seq(x, trans=True)).permute(0, 2, 1).view(B, C, W, H).transpose(2, 3)
        out_dt = torch.flip(self.mamba_dt(get_seq(x, True, True)), [1]).permute(0, 2, 1).view(B, C, W, H).transpose(2, 3)

        stack_feat = torch.cat([out_lr, out_rl, out_td, out_dt], dim=1)
        gates = self.gate_fusion(stack_feat)
        g_lr, g_rl, g_td, g_dt = torch.split(gates, C, dim=1)

        fused = out_lr * g_lr + out_rl * g_rl + out_td * g_td + out_dt * g_dt
        # Residual connection
        return self.gamma * self.proj_out(fused) + x


class GMambaBlock(nn.Module):
    """G-Mamba (Directional Geometric Mamba): geometry-guided scanning layer (L3).

    Takes the well-formed spatial features from L1/L2 and converges the
    boundaries by strictly following the DGM centripetal flow field,
    via soft geometric prompting of the four directional scans.
    """

    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super(GMambaBlock, self).__init__()
        self.mamba_lr = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_rl = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_td = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_dt = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Fusion gate additionally conditioned on the 2-channel flow (gx, gy)
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(d_model * 4 + 2, d_model * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model * 4),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.proj_out = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model), nn.ReLU(inplace=True)
        )

    def forward(self, x_base, flow_field, d_map_coarse):
        B, C, H, W = x_base.shape
        N = H * W

        flow_norm = F.normalize(flow_field, p=2, dim=1)
        gx, gy = flow_norm[:, 0:1], flow_norm[:, 1:2]
        s_lr, s_rl, s_td, s_dt = F.relu(gx), F.relu(-gx), F.relu(gy), F.relu(-gy)

        # Soft geometric prompting driven by the coarse boundary map
        t_lr = 1.0 + d_map_coarse * s_lr
        t_rl = 1.0 + d_map_coarse * s_rl
        t_td = 1.0 + d_map_coarse * s_td
        t_dt = 1.0 + d_map_coarse * s_dt

        def get_seq(feat, flip=False, trans=False):
            if trans: feat = feat.transpose(2, 3).contiguous()
            s = feat.reshape(B, C, N).permute(0, 2, 1)
            return torch.flip(s, [1]) if flip else s

        out_lr = self.mamba_lr(get_seq(x_base * t_lr)).permute(0, 2, 1).view(B, C, H, W)
        out_rl = torch.flip(self.mamba_rl(get_seq(x_base * t_rl, True)), [1]).permute(0, 2, 1).view(B, C, H, W)
        out_td = self.mamba_td(get_seq(x_base * t_td, trans=True)).permute(0, 2, 1).view(B, C, W, H).transpose(2, 3)
        out_dt = torch.flip(self.mamba_dt(get_seq(x_base * t_dt, True, True)), [1]).permute(0, 2, 1).view(B, C, W, H).transpose(2, 3)

        stack_feat = torch.cat([out_lr, out_rl, out_td, out_dt, gx, gy], dim=1)
        gates = self.gate_fusion(stack_feat)
        g_lr, g_rl, g_td, g_dt = torch.split(gates, C, dim=1)

        fused = out_lr * g_lr + out_rl * g_rl + out_td * g_td + out_dt * g_dt
        return self.gamma * self.proj_out(fused) + x_base


class CascadeGN_SSM(nn.Module):
    """Cascade Geometry-Navigated SSM: decoupled representation learning.

    L1/L2 build spatial context; L3 (G-Mamba) performs geometry-guided
    boundary alignment.
    Each layer has its own internal residual connection.
    """

    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super(CascadeGN_SSM, self).__init__()

        # 2D local prior (depthwise conv)
        self.local_2d_prior = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model), nn.SiLU(inplace=True)
        )

        # Progressive 3-layer cascade
        self.layer1 = SpatialMambaBlock(d_model, d_state, d_conv, expand)
        self.layer2 = SpatialMambaBlock(d_model, d_state, d_conv, expand)
        self.layer3 = GMambaBlock(d_model, d_state, d_conv, expand)

    def forward(self, x, flow_field, d_map_coarse):
        # 1. Local prior
        x_local = self.local_2d_prior(x)
        x_base = x + x_local

        # 2. Cascade: context -> refinement -> DGM boundary alignment
        x_l1 = self.layer1(x_base)
        x_l2 = self.layer2(x_l1)
        out_l3 = self.layer3(x_l2, flow_field, d_map_coarse)

        return out_l3


class DGMNet(nn.Module):
    """DGM-Net main network.

    Forward returns:
        training:  (out_c_up, out_dgm_up, out_aux_up)
        inference: (out_c_up, out_dgm_up)

    out_dgm channels: [V, Gx, Gy, D_logits, C_map] (see dgm_targets.py).
    """

    def __init__(self, n_classes=19):
        super(DGMNet, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT,
                                  replace_stride_with_dilation=[False, True, True])
        self.low_level_features = nn.Sequential(*list(resnet.children())[:5])
        self.backbone = nn.Sequential(*list(resnet.children())[5:-2])

        self.reduce_high = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.low_level_compress = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        # Coarse DGM prediction head ("draft sketch")
        self.dgm_predictor = nn.Sequential(
            nn.Conv2d(304, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, 1)
        )

        self.gn_ssm = CascadeGN_SSM(d_model=256)

        # D-map feedback refiner: takes post-Mamba features (+ low-level 48ch)
        # and predicts a boundary correction residual (Delta D)
        self.d_refiner = nn.Sequential(
            nn.Conv2d(256 + 48, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.offset_scaler = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(128, n_classes, 1)
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(305, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        low_raw = self.low_level_features(x)
        high = self.backbone(low_raw)
        high_r = self.reduce_high(high)
        high_up = F.interpolate(high_r, size=low_raw.shape[2:], mode='bilinear', align_corners=False)
        low_c = self.low_level_compress(low_raw)

        # 1. Coarse geometric prediction
        out_dgm = self.dgm_predictor(torch.cat([low_c, high_up], 1))
        v_map = torch.sigmoid(out_dgm[:, 0:1])
        flow = torch.tanh(out_dgm[:, 1:3])
        c_map = torch.sigmoid(out_dgm[:, 4:5])
        edge = torch.tanh(torch.sqrt(flow[:, 0:1]**2 + flow[:, 1:2]**2 + 1e-4))

        d_coarse_logits = out_dgm[:, 3:4]
        d_map_coarse = torch.sigmoid(d_coarse_logits)

        # 2. Global Mamba scan carrying the geometric draft
        flow_d = F.interpolate(flow, size=high_r.shape[2:], mode='bilinear', align_corners=False)
        d_map_coarse_d = F.interpolate(d_map_coarse, size=high_r.shape[2:], mode='bilinear', align_corners=False)
        nav_high = self.gn_ssm(high_r, flow_d, d_map_coarse_d)
        nav_up = F.interpolate(nav_high, size=low_raw.shape[2:], mode='bilinear', align_corners=False)

        out_aux = self.aux_head(nav_high)
        refine_feat = torch.cat([nav_up, low_c], dim=1)

        # Draft + correction = refined boundary logits
        delta_d = self.d_refiner(refine_feat)
        d_refined_logits = d_coarse_logits + delta_d
        d_map_refined = torch.sigmoid(d_refined_logits)

        # 3. Geometric Offset-Aligned Decoding (GOAD)
        B, C, H_low, W_low = low_c.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H_low, dtype=low_c.dtype, device=low_c.device),
            torch.linspace(-1, 1, W_low, dtype=low_c.dtype, device=low_c.device),
            indexing='ij'
        )
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        dynamic_scale = self.offset_scaler(flow) * 0.2

        # Align low-level features using the refined boundary map
        offset = flow.permute(0, 2, 3, 1) * d_map_refined.permute(0, 2, 3, 1) * dynamic_scale.permute(0, 2, 3, 1)
        aligned_grid = torch.clamp(grid + offset, -1.0, 1.0)
        aligned_low_c = F.grid_sample(low_c, aligned_grid, mode='bilinear',
                                      padding_mode='reflection', align_corners=True)

        guided_low_c = aligned_low_c * (1.0 + d_map_refined)
        guided_nav_up = nav_up * (1.0 + v_map * 0.5)

        feat = torch.cat([guided_low_c, guided_nav_up], 1)
        out_c = self.seg_head(torch.cat([feat, edge], 1))

        # Pass the *refined* D logits to the loss so gradients flow to both
        # the dgm_predictor (better drafts) and the d_refiner (better fixes)
        out_dgm_for_loss = torch.cat([v_map, flow, d_refined_logits, c_map], dim=1)
        out_dgm_up = F.interpolate(out_dgm_for_loss, size=(h, w), mode='bilinear', align_corners=False)

        # 4. Staged boundary-guided upsampling (using the refined map)
        out_c_half = F.interpolate(out_c, scale_factor=2, mode='bilinear', align_corners=False)
        d_map_half = F.interpolate(d_map_refined, scale_factor=2, mode='bilinear', align_corners=False)
        out_c_half = out_c_half + (out_c_half * d_map_half)
        out_c_up = F.interpolate(out_c_half, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            out_aux_up = F.interpolate(out_aux, size=(h, w), mode='bilinear', align_corners=False)
            return out_c_up, out_dgm_up, out_aux_up
        else:
            return out_c_up, out_dgm_up
