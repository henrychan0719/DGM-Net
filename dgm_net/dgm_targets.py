"""Adaptive DGM target generation.

Given a semantic label map, produces the 5-channel geometric supervision
used by DGM-Net:

    1. ID-agnostic dynamic geometric radius (works on any dataset)
    2. Centripetal Convergence Field (CCF)
    3. Curvature awareness (C_map)
    4. High-fidelity morphological boundary map D (preserves thin
       structures such as poles / fences)

Output shape: (5, H, W) with channels [V, Gx, Gy, D, C_map].
"""

import numpy as np
import cv2


def compute_dgm_targets_adaptive(mask_np, ignore_index=255):
    H, W = mask_np.shape
    valid = (mask_np != ignore_index)
    if not np.any(valid): return np.zeros((5, H, W), dtype=np.float32)

    # 1. Boundary detection
    boundary = np.zeros_like(mask_np, dtype=bool)
    boundary[:, 1:] |= valid[:, 1:] & valid[:, :-1] & (mask_np[:, 1:] != mask_np[:, :-1])
    boundary[1:, :] |= valid[1:, :] & valid[:-1, :] & (mask_np[1:, :] != mask_np[:-1, :])

    # 2. Distance transform
    mask_for_dist = (~boundary & valid).astype(np.uint8)
    dist = cv2.distanceTransform(mask_for_dist, cv2.DIST_L2, 5)

    # ==========================================
    # 3. Dynamic R matrix from physical geometry (fully ID-agnostic)
    # ==========================================
    R_matrix = np.full((H, W), 8.0, dtype=np.float32)
    unique_classes = np.unique(mask_np[valid])

    for cid in unique_classes:
        class_mask = (mask_np == cid).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(class_mask, connectivity=8)

        for label_id in range(1, num_labels):
            instance_mask = (labels == label_id)

            # Geometric features of the connected component
            max_dist = dist[instance_mask].max()
            area = instance_mask.sum()
            perimeter = (boundary & instance_mask).sum()

            # Thinness index
            thinness = perimeter / (np.sqrt(area) + 1e-6)

            # Dynamic R range assignment
            if thinness > 12.0 and max_dist < 25.0:
                r_min, r_max = 2.0, 20.0     # very thin objects (fence / pole)
            elif max_dist > 40.0 and thinness < 10.0:
                r_min, r_max = 15.0, 150.0   # large structures (wall / building)
            else:
                r_min, r_max = 6.0, 100.0    # regular objects

            dynamic_r = np.clip(max_dist, r_min, r_max)
            R_matrix[instance_mask] = dynamic_r

    # ==========================================
    # 4. Centripetal Convergence Field (CCF)
    # ==========================================
    V = np.zeros((H, W), dtype=np.float32)
    V[valid] = np.clip(dist[valid] / R_matrix[valid], 0.0, 1.0)
    V[boundary] = 0.0

    Vg = V.copy(); Vg[~valid] = 0
    Gx = cv2.Sobel(Vg, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(Vg, cv2.CV_32F, 0, 1, ksize=3)

    # ==========================================
    # 5. Geometric curvature awareness (Laplacian)
    # ==========================================
    C_map = cv2.Laplacian(Vg, cv2.CV_32F, ksize=3)
    C_map = np.abs(C_map)
    if C_map.max() > 0:
        C_map /= C_map.max()

    # ==========================================
    # 6. High-fidelity D map (morphological gradient, thin-structure safe)
    # ==========================================
    C_u = mask_np.copy()
    C_u[~valid] = 0
    C_u_uint8 = C_u.astype(np.uint8)

    # Cross-shaped kernel protects thin vertical / horizontal features
    k_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gradient = cv2.morphologyEx(C_u_uint8, cv2.MORPH_GRADIENT, k_cross)

    D = np.zeros_like(V, dtype=np.float32)
    D[gradient > 0] = 1.0

    # Tight Gaussian blur to avoid boundary halos eating into the background
    D = cv2.GaussianBlur(D, (3, 3), 0.5)
    if D.max() > 0:
        D /= D.max()

    # ==========================================
    # Clean invalid regions and output
    # ==========================================
    V[~valid] = 0; Gx[~valid] = 0; Gy[~valid] = 0; D[~valid] = 0; C_map[~valid] = 0
    return np.stack([V, Gx, Gy, D, C_map], axis=0)
