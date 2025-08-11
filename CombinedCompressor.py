"""
This code is adapted from SparseGPT: https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py
Modified on: 2025-07-30
Modified by: Seokho Han

Kaggle-friendly logging:
- Uses tqdm progress bars (single cell-friendly)
- Structured status lines via pbar.set_postfix / tqdm.write
- No external file logging; output cell only
"""

import math
import time
import torch
import torch.nn as nn
import transformers
from Quantizer import *
from tqdm.auto import tqdm
from pathlib import Path
from threading import Lock

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


_GLOBAL_PBAR = None
_GLOBAL_PBAR_LOCK = Lock()
_ASCII_PRINTED = False


def _print_ascii_once(force: bool = False):
    global _ASCII_PRINTED
    if _ASCII_PRINTED and not force:
        return
    try:
        ascii_path = Path(__file__).with_name("ascii-art.txt")
        if ascii_path.exists():
            art = ascii_path.read_text(encoding="utf-8", errors="ignore").rstrip("\n")
            # Print above the progress bar so it stays at the top
            print(art)
    except Exception:
        # If ascii cannot be loaded, silently continue
        pass
    _ASCII_PRINTED = True


class _GlobalProgress:
    """A single global tqdm progress bar shared across the whole compression run.

    - The total is dynamic. Each layer can extend the total by calling add_total().
    - Messages are printed below the bar via tqdm.write().
    """

    def __init__(self):
        _print_ascii_once()
        self.pbar = tqdm(
            total=0,
            bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=True,
            dynamic_ncols=False,
            ncols=400,
            position=0,
            mininterval=0.1,
        )

    def add_total(self, n: int):
        if n <= 0:
            return
        # Dynamically expand total
        new_total = (self.pbar.total or 0) + int(n)
        self.pbar.total = new_total
        self.pbar.refresh()

    def set_description(self, text: str):
        # Force refresh so Jupyter/Kaggle display updates immediately
        self.pbar.set_description_str(text, refresh=True)

    def set_postfix(self, postfix: dict):
        self.pbar.set_postfix(postfix, refresh=True)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def note(self, msg: str):
        tqdm.write(msg)

    def close(self):
        self.pbar.close()

    def reset_total(self, total: int):
        # Reset the bar to represent total number of layers
        total = max(int(total), 0)
        self.pbar.total = total
        self.pbar.n = 0
        self.pbar.refresh()


def _get_global_progress() -> _GlobalProgress:
    global _GLOBAL_PBAR
    with _GLOBAL_PBAR_LOCK:
        if _GLOBAL_PBAR is None:
            _GLOBAL_PBAR = _GlobalProgress()
        return _GLOBAL_PBAR


def close_global_progress():
    """Optionally close the global progress bar when the entire run is finished."""
    global _GLOBAL_PBAR
    with _GLOBAL_PBAR_LOCK:
        if _GLOBAL_PBAR is not None:
            _GLOBAL_PBAR.close()
            _GLOBAL_PBAR = None


def start_global_progress(total_layers: int, desc: str | None = None):
    """Initialize the single global progress bar with the total number of layers.

    Call this once before running pruning/quantization across multiple layers.
    """
    global _GLOBAL_PBAR
    with _GLOBAL_PBAR_LOCK:
        # Close previous bar if any, so we cleanly recreate for a new run
        if _GLOBAL_PBAR is not None:
            _GLOBAL_PBAR.close()
            _GLOBAL_PBAR = None
        # Force print ASCII at the start of each run
        _print_ascii_once(force=True)
        # Create a fresh global bar
        _GLOBAL_PBAR = _GlobalProgress()
        _GLOBAL_PBAR.reset_total(total_layers)
        if desc:
            _GLOBAL_PBAR.set_description(desc)
        return _GLOBAL_PBAR


class CombinedCompressor:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        elif isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)              # (B, Ckhkw, L)
            inp = inp.permute([1, 0, 2])  # (Ckhkw, B, L)
            inp = inp.flatten(1)          # (Ckhkw, B*L)

        elif len(inp.shape) == 4:
            inp = inp.view(inp.size(0), -1)

        # EMA-style update of H
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # Follow SparseGPT-style inversion path
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        # --- Global tqdm: layer-based overall progress (0..#layers) ---
        gbar = _get_global_progress()
        layer_desc = f"{self.layer.__class__.__name__}({self.rows}x{self.columns})"
        gbar.note(
            f"[Start] {layer_desc} | blocksize={blocksize}, sparsity={sparsity:.3f}, "
            f"prunen={prunen}, prunem={prunem}, percdamp={percdamp}"
        )

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            # Keep single global bar; description is controlled by caller

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Build/derive mask1; track approximate mask ratio for info if needed
            mask_ratio = None
            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # threshold by global sparsity within this block
                    thresh = torch.sort(tmp.flatten())[0][int(max(0, min(tmp.numel() - 1, tmp.numel() * sparsity)))]
                    mask1 = tmp <= thresh
                    mask_ratio = mask1.float().mean().item()
            else:
                mask1 = torch.zeros_like(W1, dtype=torch.bool)

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    # keep prunen smallest within the window
                    idx = torch.topk(tmp, prunen, dim=1, largest=False)[1]
                    mask1.scatter_(1, i + idx, True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            block_loss = torch.sum(Losses1, 1) / 2
            Losses += block_loss

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                dbg_err = torch.sum((self.layer(self.inp1) - self.out1) ** 2).item()
                dbg_loss = torch.sum(Losses).item()
                gbar.note(f"[DEBUG] layer_err={dbg_err:.6e} | cum_loss={dbg_loss:.6e}")

            # single global bar is layer-based; do not advance per-block

        torch.cuda.synchronize()
        total_time = time.time() - tick
        total_err = torch.sum(Losses).item()
        # Keep the global bar and print layer-wise summary beneath it
        gbar.note(f"[Layer Done] {layer_desc} | time {total_time:.2f}s | err {total_err:.6e}")
        # advance by one completed layer
        gbar.update(1)

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            dbg_err_final = torch.sum((self.layer(self.inp1) - self.out1) ** 2).item()
            tqdm.write(f"[DEBUG] final layer_err={dbg_err_final:.6e}")

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
