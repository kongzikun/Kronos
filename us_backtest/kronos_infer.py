from typing import Tuple, Optional
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


def load_model(device: torch.device,
               tokenizer_path: Optional[str] = None,
               model_path: Optional[str] = None) -> Tuple[KronosTokenizer, Kronos]:
    """Load pretrained Kronos tokenizer and model.

    If weights are not available locally (HuggingFace cache or repo path), prints
    a clear instruction and exits to avoid using random weights.
    """
    tok_src = os.environ.get("KRONOS_TOKENIZER_PATH") or tokenizer_path or "NeoQuasar/Kronos-Tokenizer-base"
    mdl_src = os.environ.get("KRONOS_MODEL_PATH") or model_path or "NeoQuasar/Kronos-base"
    try:
        tokenizer = KronosTokenizer.from_pretrained(tok_src)
        model = Kronos.from_pretrained(mdl_src)
    except Exception as e:
        print(
            "[Kronos] Failed to load pretrained weights.\n"
            f"Tried tokenizer: {tok_src}\nTried model: {mdl_src}\n"
            "Options:\n"
            "  - Set env KRONOS_TOKENIZER_PATH / KRONOS_MODEL_PATH to local directories, or\n"
            "  - Pass --tokenizer_path / --model_path via CLI (run_kronos_us), or\n"
            "  - Ensure network access to Hugging Face repositories.\n"
            f"Underlying error: {e}"
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
        try:
            from torch.amp import autocast as _autocast
            amp_ctx = lambda enabled: _autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=enabled)
        except Exception:
            # Fallback to legacy namespaces if available
            amp_ctx = (torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast)
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
