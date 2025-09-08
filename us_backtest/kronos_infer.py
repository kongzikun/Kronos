from typing import Tuple

import numpy as np
import torch
import os
from huggingface_hub import hf_hub_download

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference


def load_model(device: torch.device) -> Tuple[KronosTokenizer, Kronos]:
    """Load pretrained Kronos tokenizer and model."""
    tokenizer_repo = "NeoQuasar/Kronos-Tokenizer-base"
    model_repo = "NeoQuasar/Kronos-base"
    token = os.environ.get("HF_TOKEN")
    try:
        tok_weights = hf_hub_download(repo_id=tokenizer_repo, filename="model.safetensors", token=token)
        hf_hub_download(repo_id=tokenizer_repo, filename="config.json", token=token)
        mdl_weights = hf_hub_download(repo_id=model_repo, filename="model.safetensors", token=token)
        hf_hub_download(repo_id=model_repo, filename="config.json", token=token)
    except Exception as e:
        raise RuntimeError(
            "Failed to download Kronos pretrained weights. Please run `huggingface-cli login` or download the files manually."
        ) from e

    tok_cache_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(tok_weights))))
    mdl_cache_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(mdl_weights))))

    tokenizer = KronosTokenizer.from_pretrained(
        tokenizer_repo, cache_dir=tok_cache_dir, local_files_only=True, token=token
    )
    model = Kronos.from_pretrained(
        model_repo, cache_dir=mdl_cache_dir, local_files_only=True, token=token
    )

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
