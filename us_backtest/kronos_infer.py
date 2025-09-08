from typing import Tuple
import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path so `model` can be imported when running as a script
_CUR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_CUR)
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Import directly from implementation module to access helper functions
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


def load_model(device: torch.device) -> Tuple[KronosTokenizer, Kronos]:
    """Load pretrained Kronos tokenizer and model.

    If weights are not available locally (HuggingFace cache or repo path), prints
    a clear instruction and exits to avoid using random weights.
    """
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    except Exception:
        print(
            "[Kronos] Pretrained weights not found.\n"
            "Please download: \n"
            "  - Tokenizer: NeoQuasar/Kronos-Tokenizer-base\n"
            "  - Model:     NeoQuasar/Kronos-base\n"
            "From: https://huggingface.co/NeoQuasar/ \n"
            "And ensure they are loadable by KronosTokenizer.from_pretrained / Kronos.from_pretrained."
        )
        raise SystemExit(1)

    tokenizer.eval()
    model.eval()
    tokenizer.to(device)
    model.to(device)
    return tokenizer, model


def predict_batch(
    tokenizer: KronosTokenizer,
    model: Kronos,
    x: np.ndarray,
    x_stamp: np.ndarray,
    y_stamp: np.ndarray,
    device: torch.device,
    samples: int,
    T: float,
    top_p: float,
) -> np.ndarray:
    """Predict future sequences for a batch of windows.

    Returns the de-normalized predictions for the next H steps only (not the
    entire context), shape: [batch, H, features].
    """
    # Per-window z-score fit (no leakage) then clip
    x_mean = x.mean(axis=1, keepdims=True)
    x_std = x.std(axis=1, keepdims=True)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -5, 5)

    x_tensor = torch.from_numpy(x_norm.astype(np.float32)).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp.astype(np.float32)).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp.astype(np.float32)).to(device)

    H = y_stamp.shape[1]
    with torch.no_grad():
        # Optional AMP for throughput; safe as we de-norm on CPU/NumPy
        amp_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast
        with amp_ctx(enabled=False):
            preds_full = auto_regressive_inference(
                tokenizer,
                model,
                x_tensor,
                x_stamp_tensor,
                y_stamp_tensor,
                max_context=512,
                pred_len=H,
                clip=5,
                T=T,
                top_k=0,
                top_p=top_p,
                sample_count=samples,
                verbose=False,
            )

    # Slice to only the last H timesteps per design of the inference util
    preds = preds_full[:, -H:, :]
    preds = preds * (x_std + 1e-5) + x_mean
    return preds
