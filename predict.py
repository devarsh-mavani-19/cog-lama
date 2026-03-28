"""Cog wrapper for LaMa (Large Mask Inpainting) model.

Loads the pre-trained big-lama checkpoint and exposes image inpainting
as a Cog-compatible prediction endpoint.
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from cog import BasePredictor, Input, Path

# ---------------------------------------------------------------------------
# Paths inside the container (populated by cog.yaml `run` commands)
# ---------------------------------------------------------------------------
LAMA_ROOT = "/lama"
MODEL_DIR = os.path.join(LAMA_ROOT, "big-lama")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "models", "best.ckpt")
TRAIN_CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")


def _add_lama_to_path():
    """Ensure the cloned LaMa repo is importable."""
    if LAMA_ROOT not in sys.path:
        sys.path.insert(0, LAMA_ROOT)
    os.environ.setdefault("TORCH_HOME", LAMA_ROOT)


def _load_image(path: str) -> np.ndarray:
    """Load an image as a float32 RGB array in [0, 1] with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def _load_mask(path: str) -> np.ndarray:
    """Load a mask as a float32 array in [0, 1] with shape (H, W, 1).

    Any non-zero pixel is treated as the region to inpaint.
    """
    mask = Image.open(path).convert("L")
    mask = np.array(mask).astype(np.float32) / 255.0
    # Binarize: anything above 0.5 becomes 1
    mask = (mask > 0.5).astype(np.float32)
    return mask[:, :, np.newaxis]


def _pad_to_modulo(img: np.ndarray, mod: int) -> tuple:
    """Pad image so that H and W are divisible by *mod*.

    Returns the padded image and (pad_h, pad_w) used.
    """
    h, w = img.shape[:2]
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    if img.ndim == 3:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="symmetric")
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="symmetric")
    return padded, (pad_h, pad_w)


class Predictor(BasePredictor):
    """Cog predictor that wraps the LaMa inpainting model."""

    def setup(self):
        """Load the LaMa model and checkpoint into GPU memory."""
        _add_lama_to_path()

        from saicinpainting.training.trainers import load_checkpoint

        # Read the training config that ships with the checkpoint.
        # OmegaConf resolves ${...} variable interpolations in the YAML.
        train_config = OmegaConf.load(TRAIN_CONFIG_PATH)
        train_config.training_model.predict_only = True

        self.model = load_checkpoint(
            train_config, CHECKPOINT_PATH, map_location="cuda", strict=False
        )
        self.model.eval()
        self.model.freeze()
        self.device = torch.device("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image to inpaint"),
        mask: Path = Input(
            description="Mask image (white = area to inpaint, black = keep)"
        ),
    ) -> Path:
        """Run LaMa inpainting on the given image + mask pair."""

        # ------------------------------------------------------------------
        # 1. Load & validate inputs
        # ------------------------------------------------------------------
        img = _load_image(str(image))  # (H, W, 3) float32 [0, 1]
        msk = _load_mask(str(mask))  # (H, W, 1) float32 {0, 1}

        # Resize mask to match image if dimensions differ
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(
                msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            if msk.ndim == 2:
                msk = msk[:, :, np.newaxis]

        orig_h, orig_w = img.shape[:2]

        # ------------------------------------------------------------------
        # 2. Pad so dimensions are divisible by 8 (required by the network)
        # ------------------------------------------------------------------
        img_padded, (pad_h, pad_w) = _pad_to_modulo(img, 8)
        msk_padded, _ = _pad_to_modulo(msk, 8)

        # ------------------------------------------------------------------
        # 3. Build batch tensor  (N, C, H, W)
        # ------------------------------------------------------------------
        img_tensor = (
            torch.from_numpy(img_padded)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        msk_tensor = (
            torch.from_numpy(msk_padded)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        batch = {"image": img_tensor, "mask": msk_tensor}

        # ------------------------------------------------------------------
        # 4. Run inference
        # ------------------------------------------------------------------
        with torch.no_grad():
            batch = self.model(batch)
            result = batch["inpainted"]  # (1, 3, H, W) float [0, 1]

        # ------------------------------------------------------------------
        # 5. Post-process: remove padding, convert to uint8 image
        # ------------------------------------------------------------------
        result = result[0].permute(1, 2, 0).cpu().numpy()  # (H_pad, W_pad, 3)
        result = result[:orig_h, :orig_w, :]  # crop padding
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # 6. Save & return
        # ------------------------------------------------------------------
        out_path = os.path.join(tempfile.mkdtemp(), "output.png")
        Image.fromarray(result).save(out_path)
        return Path(out_path)
