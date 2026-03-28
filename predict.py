"""Cog wrapper for LaMa (Large Mask Inpainting) model.

Uses the JIT-traced (TorchScript) big-lama generator for clean inpainting.
"""

import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image

from cog import BasePredictor, Input, Path

# ---------------------------------------------------------------------------
# Path to the JIT model inside the container (downloaded in cog.yaml)
# ---------------------------------------------------------------------------
LAMA_MODEL_PATH = "/lama/big-lama.pt"


def _norm_img(np_img: np.ndarray) -> np.ndarray:
    """Normalize image: add channel dim if needed, HWC->CHW, scale to [0,1]."""
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))  # HWC -> CHW
    np_img = np_img.astype("float32") / 255
    return np_img


def _load_image(path: str) -> np.ndarray:
    """Load an image as a uint8 RGB array with shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_mask(path: str) -> np.ndarray:
    """Load a mask as a uint8 grayscale array with shape (H, W).

    Any non-zero pixel is treated as the region to inpaint.
    """
    mask = Image.open(path).convert("L")
    return np.array(mask)


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
        """Load the JIT-traced LaMa generator into GPU memory."""
        self.device = torch.device("cuda")
        self.model = (
            torch.jit.load(LAMA_MODEL_PATH, map_location="cpu")
            .to(self.device)
            .eval()
        )

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
        img = _load_image(str(image))   # (H, W, 3) uint8
        msk = _load_mask(str(mask))     # (H, W)    uint8

        # Resize mask to match image if dimensions differ
        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(
                msk, (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        orig_h, orig_w = img.shape[:2]

        # ------------------------------------------------------------------
        # 2. Pad so dimensions are divisible by 8 (required by the network)
        # ------------------------------------------------------------------
        img_padded, (pad_h, pad_w) = _pad_to_modulo(img, 8)
        msk_padded, _ = _pad_to_modulo(msk, 8)

        # ------------------------------------------------------------------
        # 3. Normalize and build tensors (N, C, H, W)
        # ------------------------------------------------------------------
        img_norm = _norm_img(img_padded)           # (3, H, W) float32 [0,1]
        msk_norm = _norm_img(msk_padded)           # (1, H, W) float32 [0,1]

        # Binarize mask: anything > 0 becomes 1
        msk_norm = (msk_norm > 0) * 1.0

        img_tensor = (
            torch.from_numpy(img_norm).unsqueeze(0).to(self.device)
        )  # (1, 3, H, W)
        msk_tensor = (
            torch.from_numpy(msk_norm).unsqueeze(0).to(self.device)
        )  # (1, 1, H, W)

        # ------------------------------------------------------------------
        # 4. Run inference with the JIT model
        # ------------------------------------------------------------------
        with torch.no_grad():
            result = self.model(img_tensor, msk_tensor)  # (1, 3, H, W)

        # ------------------------------------------------------------------
        # 5. Post-process: remove padding, convert to uint8 image
        # ------------------------------------------------------------------
        result = result[0].permute(1, 2, 0).cpu().numpy()  # (H_pad, W_pad, 3)
        result = result[:orig_h, :orig_w, :]                # crop padding
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # 6. Save & return
        # ------------------------------------------------------------------
        out_path = os.path.join(tempfile.mkdtemp(), "output.png")
        Image.fromarray(result).save(out_path)
        return Path(out_path)
