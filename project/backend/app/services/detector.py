from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import io

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import cv2

import segmentation_models_pytorch as smp


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class DetectorConfig:
    classifier_path: str = "models/best_classifier.pth"
    segmenter_path: str = "models/best_ship_segmenter.pth"

    device: str = "cpu"

    # Gate threshold (as you described)
    classifier_threshold: float = 0.05

    # Segmentation threshold
    mask_threshold: float = 0.5

    # Remove tiny blobs
    min_blob_area_px: int = 50

    # Return at most this many boxes (optional safety)
    max_boxes: int = 50


class ShipDetector:
    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()
        self.device = torch.device(self.cfg.device)
        self.classifier: Optional[nn.Module] = None
        self.segmenter: Optional[nn.Module] = None

    def load(self) -> None:
        self.classifier = self._load_classifier(self.cfg.classifier_path).to(self.device).eval()
        self.segmenter = self._load_segmenter(self.cfg.segmenter_path).to(self.device).eval()
        print("[ShipDetector] Loaded classifier + segmenter")

    @torch.no_grad()
    def detect(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        if self.classifier is None or self.segmenter is None:
            raise RuntimeError("Models not loaded")

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = pil.size

        # --- Stage 1: classifier gate ---
        x_cls = self._preprocess(pil, size=384)  # [1,3,384,384]
        prob = self._classifier_prob(x_cls)      # float in [0,1]

        if prob < self.cfg.classifier_threshold:
            # Early exit (no ships)
            return []

        # --- Stage 2: segmenter ---
        x_seg = self._preprocess(pil, size=768)  # [1,3,768,768]
        mask_prob = self._segmenter_mask(x_seg)  # [768,768] float

        # Threshold + cleanup + boxes in SEGMENTER space (768x768)
        boxes_768 = mask_to_boxes(
            mask_prob,
            thr=self.cfg.mask_threshold,
            min_area=self.cfg.min_blob_area_px,
            max_boxes=self.cfg.max_boxes,
        )

        # Scale boxes from 768-space back to ORIGINAL image pixel coords
        sx = orig_w / 768.0
        sy = orig_h / 768.0

        preds: List[Dict[str, Any]] = []
        for (x, y, w, h, score) in boxes_768:
            preds.append({
                "x": float(x * sx),
                "y": float(y * sy),
                "w": float(w * sx),
                "h": float(h * sy),
                "score": float(score),
                "label": "ship",
                # optional: include classifier prob for debugging
                "gate_prob": float(prob),
            })

        return preds

    # ----------------- model loading -----------------

    def _load_classifier(self, path: str) -> nn.Module:
        """
        ResNet34 binary classifier.
        Expects either:
        - full model saved (torch.save(model))
        - state_dict saved (torch.save(model.state_dict()))
        - checkpoint dict containing state_dict under common keys
        """
        model = torchvision.models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)  # single logit

        ckpt = torch.load(path, map_location="cpu")
        state = extract_state_dict(ckpt)

        if state is not None:
            state = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state.items()
            }

        if state is None and isinstance(ckpt, nn.Module):
            return ckpt

        if state is None:
            raise RuntimeError(f"Could not parse classifier weights: {path}")

        model.load_state_dict(state, strict=True)
        return model

    def _load_segmenter(self, path: str) -> nn.Module:
        """
        U-Net with EfficientNet-B4 encoder.
        Output: 1 channel logits (we apply sigmoid).
        """
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        ckpt = torch.load(path, map_location="cpu")
        state = extract_state_dict(ckpt)

        if state is not None:
            state = {
                k.replace("model.", "", 1) if k.startswith("model.") else k: v
                for k, v in state.items()
            }

        if state is None and isinstance(ckpt, nn.Module):
            return ckpt

        if state is None:
            raise RuntimeError(f"Could not parse segmenter weights: {path}")

        model.load_state_dict(state, strict=True)
        return model

    # ----------------- forward helpers -----------------

    def _preprocess(self, pil: Image.Image, size: int) -> torch.Tensor:
        # resize to (size,size), convert to tensor [1,3,H,W], normalize ImageNet
        img = pil.resize((size, size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.to(self.device)

    def _classifier_prob(self, x: torch.Tensor) -> float:
        # model returns logits; apply sigmoid
        logits = self.classifier(x)  # [1,1] (most likely)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        p = torch.sigmoid(logits).flatten()[0].item()
        return float(p)

    def _segmenter_mask(self, x: torch.Tensor) -> np.ndarray:
        # returns per-pixel probabilities [768,768]
        logits = self.segmenter(x)  # [1,1,768,768]
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
        return prob


def extract_state_dict(ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "model_state"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes the dict itself is state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return None


def mask_to_boxes(
    mask_prob: np.ndarray,
    thr: float,
    min_area: int,
    max_boxes: int,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Input: mask_prob [H,W] float in [0,1]
    Output: list of (x,y,w,h,score) in the same HxW coordinate system
    """
    H, W = mask_prob.shape
    bin_mask = (mask_prob >= thr).astype(np.uint8)

    # connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    boxes: List[Tuple[int, int, int, int, float]] = []
    # label 0 is background
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        if area < min_area:
            continue

        # score: mean prob in the blob
        blob = (labels == lab)
        score = float(mask_prob[blob].mean()) if blob.any() else 0.0

        boxes.append((int(x), int(y), int(w), int(h), score))

    # sort by score descending
    boxes.sort(key=lambda t: t[4], reverse=True)
    return boxes[:max_boxes]
