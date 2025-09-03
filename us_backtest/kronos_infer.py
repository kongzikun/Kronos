from typing import Tuple

import numpy as np
import torch

from model import Kronos, KronosTokenizer, auto_regressive_inference


def load_model(device: torch.device) -> Tuple[KronosTokenizer, Kronos]:
    """Load pretrained Kronos tokenizer and model."""
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    except Exception as e:
        print(
            "Pretrained Kronos weights not found. Please download from https://huggingface.co/NeoQuasar/ and place them so that KronosTokenizer.from_pretrained and Kronos.from_pretrained can load them.")
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
    """Predict future sequences for a batch of windows."""
    x_mean = x.mean(axis=1, keepdims=True)
    x_std = x.std(axis=1, keepdims=True)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -5, 5)

    x_tensor = torch.from_numpy(x_norm.astype(np.float32)).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp.astype(np.float32)).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp.astype(np.float32)).to(device)

    preds = auto_regressive_inference(
        tokenizer,
        model,
        x_tensor,
        x_stamp_tensor,
        y_stamp_tensor,
        max_context=512,
        pred_len=y_stamp.shape[1],
        clip=5,
        T=T,
        top_k=0,
        top_p=top_p,
        sample_count=samples,
        verbose=False,
    )

    preds = preds * (x_std + 1e-5) + x_mean
    return preds
