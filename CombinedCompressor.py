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

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class _KaggleLogger:
    """Lightweight pretty logger for Kaggle notebooks (output cell only)."""
    def __init__(self, total_blocks: int, columns: int, blocksize: int):
        self.total_blocks = total_blocks
        self.columns = columns
        self.blocksize = blocksize
        self.pbar = tqdm(total=total_blocks, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        self._last_block_start_time = None
        self._loss_running = 0.0

    def start_block(self, i1: int, i2: int):
        self._last_block_start_time = time.time()
        self.pbar.set_description_str(f"Pruning blocks ({i1}:{i2})")

    def end_block(self, losses_added: float, mask_ratio: float | None = None):
        block_time = time.time() - self._last_block_start_time if self._last_block_start_time else 0.0
        self._loss_running += float(losses_added)
        postfix = {
            "Δt(s)": f"{block_time:.2f}",
            "ΣLoss": f"{self._loss_running:.3e}",
        }
        if mask_ratio is not None:
            postfix["mask%"] = f"{100.0 * mask_ratio:.1f}"
        self.pbar.set_postfix(postfix, refresh=True)
        self.pbar.update(1)

    def note(self, msg: str):
        tqdm.write(msg)

    def done(self, total_time: float, total_loss: float):
        self.pbar.close()
        tqdm.write(f"[Done] time {total_time:.2f}s | total error {total_loss:.6e}")


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

        # --- Kaggle-friendly logging setup ---
        total_blocks = (self.columns + blocksize - 1) // blocksize
        logger = _KaggleLogger(total_blocks=total_blocks, columns=self.columns, blocksize=blocksize)
        logger.note(f"[Start] columns={self.columns}, rows={self.rows}, blocksize={blocksize}, "
                    f"sparsity={sparsity:.3f}, prunen={prunen}, prunem={prunem}, percdamp={percdamp}")

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            logger.start_block(i1, i2)

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Build/derive mask1 and log approximate mask ratio (for progress)
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
                logger.note(f"[DEBUG] layer_err={dbg_err:.6e} | cum_loss={dbg_loss:.6e}")

            # update pretty progress for this block
            logger.end_block(losses_added=float(torch.sum(block_loss).item()),
                             mask_ratio=mask_ratio)

        torch.cuda.synchronize()
        total_time = time.time() - tick
        total_err = torch.sum(Losses).item()
        logger.done(total_time=total_time, total_loss=total_err)

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
