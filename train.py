# train_llvm_gpt.py

"""
Train a tiny GPT-like transformer to predict the integer output of
randomly generated LLVM IR functions.

Dependencies:
    pip install torch matplotlib llvmlite rich

Usage:
    python train_llvm_gpt.py --target-params 1000 --epochs 30 --lr 3e-4
    python train_llvm_gpt.py --help

Controls (while training):
    Ctrl+Up     Add 10 epochs
    Ctrl+Down   Remove 10 epochs (minimum: current epoch)
"""

# ════════════════════════════════════════════════════════════════════════════
# 0.  ARGPARSE
# ════════════════════════════════════════════════════════════════════════════

run_dir = None

import json
import math
import os
import random
import signal
import sys
import time
import ctypes
import ctypes.util
from typing import List, Tuple, Optional, Dict
import numpy as np

import os as _os

_stderr_fd = _os.dup(2)
_devnull = _os.open(_os.devnull, _os.O_WRONLY)

def _suppress_c_stderr():
    _os.dup2(_devnull, 2)

def _restore_c_stderr():
    _os.dup2(_stderr_fd, 2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders
from torchinfo import summary

from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich import box

from random_llvm_gen import generate_random_function, list_supported_ops
from generate_samples import generate_example_samples
import subprocess
from run_logger import RunLogger
from csv_logger import CSVTrainingLogger
import threading
import tty
import termios
import select
import matplotlib
import matplotlib.pyplot as plt
import tkinter
from ripser import ripser
try:
    from gudhi.representations import Landscape as _Landscape
    _HAS_GUDHI = True
except ImportError:
    _HAS_GUDHI = False
from persim import wasserstein

import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

csv_log = None

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}

SCHEDULERS = {
    "cosine": lambda opt, ep: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep),
    "step": lambda opt, ep: torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, ep // 3), gamma=0.5),
    "plateau": lambda opt, ep: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5),
    "none": lambda opt, ep: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0),
    "warmup_cosine": None,
}

import argparse

import os as _os
import re
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_RE_INT = re.compile(r'^-?\d+$')
_RE_FLOAT = re.compile(r'^-?\d+\.\d+$')
_RE_LEADING_INT = re.compile(r'^(\s*-?\d+)(.*)')
_RE_PADDED_INT = re.compile(r'^\s*(-?\d+)\s*$')
_RE_PARTIAL_NUMERIC = re.compile(r'^-?\d')
_RE_LEADING_FLOAT = re.compile(r'^(\s*-?\d+\.\d+)(.*)')

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a tiny GPT on randomly generated LLVM IR functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data Generation")
    g.add_argument("--batches-per-epoch", type=int, default=150,
                   help="Training batches per epoch")
    g.add_argument("--val-batches", type=int, default=20,
                   help="Validation batches per epoch")
    g.add_argument("--max-params", type=int, default=3,
                   help="Max function parameters")
    g.add_argument("--max-ops", type=int, default=4,
                   help="Max operations in random DAG")
    g.add_argument("--allowed-ops", type=str, default="add,sub,mul",
                   help="Comma-separated LLVM ops")
    g.add_argument("--param-min", type=int, default=-20,
                   help="Min random parameter value")
    g.add_argument("--param-max", type=int, default=20,
                   help="Max random parameter value")

    g = p.add_argument_group("Model Architecture")
    g.add_argument("--target-params", type=int, default=1_000,
                   help="Target parameter count (auto config)")
    g.add_argument("--d-model", type=int, default=0,
                   help="Model dimension (0 = auto)")
    g.add_argument("--n-heads", type=int, default=0,
                   help="Attention heads (0 = auto)")
    g.add_argument("--n-layers", type=int, default=0,
                   help="Transformer layers (0 = auto)")
    g.add_argument("--max-seq-len", type=int, default=2048,
                   help="Maximum sequence length")
    g.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")
    g.add_argument("--value-loss-alpha", type=float, default=0.1,
                   help="Weight for the value regression (absolute difference) loss term. "
                   "0 = disabled, higher = more emphasis on numeric accuracy.")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=30,
                   help="Number of training epochs")
    g.add_argument("--batch-size", type=int, default=32,
                   help="Batch size")
    g.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    g.add_argument("--weight-decay", type=float, default=1e-2,
                   help="Weight decay")
    g.add_argument("--momentum", type=float, default=0.9,
                   help="Momentum (SGD only)")
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping norm")
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=list(OPTIMIZERS.keys()),
                   help="Optimizer")
    g.add_argument("--scheduler", type=str, default="none",
                   choices=list(SCHEDULERS.keys()),
                   help="LR scheduler")
    g.add_argument("--resume", type=str, default=None,
                   help="Path to a .pt checkpoint to resume training from "
                   "(e.g. llvm_gpt_model/model_epoch_15.pt)")
    g.add_argument("--structure-loss-alpha", type=float, default=0.5,
                   help="Weight for the structure-aware penalty loss term. "
                   "0 = disabled, higher = more emphasis on producing valid numbers.")
    g.add_argument("--length-loss-alpha", type=float, default=0.3,
                   help="Weight for the length penalty loss term. "
                   "0 = disabled, higher = more emphasis on correct answer length.")

    g = p.add_argument_group("Tokenizer")
    g.add_argument("--tokenizer_initial_nr", type=int, default=1000,
                   help="Number of examples generated to initialize a reasonable tokenizer")
    g.add_argument("--use-bpe", action="store_true", default=False,
                   help="Use BPE tokenizer instead of character-level")
    g.add_argument("--bpe-vocab-size", type=int, default=512,
                   help="BPE vocabulary size (only used with --use-bpe)")

    g = p.add_argument_group("Infrastructure")
    g.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device")
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    g.add_argument("--dont-plot", action="store_true", default=False,
                   help="Disable live matplotlib")
    g.add_argument("--no-plot", action="store_false", dest="plot",
                   help="Disable live plotting")
    g.add_argument("--plot-every", type=int, default=5,
                   help="Update plot every N batches")
    g.add_argument("--save-every", type=int, default=0,
                   help="Checkpoint every N epochs (0 = off)")
    g.add_argument("--no-plot-window", action="store_true", default=False,
                   help="Suppress the matplotlib window but still write plots to file")
    g.add_argument("--plot-file", type=str, default="training_plot.png",
                   help="Filename for the saved plot image (written every epoch)")
    g.add_argument("--run-dir", type=str, default="runs",
                   help="Base directory for run logs")
    g.add_argument("--log-samples", type=int, default=5,
                   help="Number of example sentences to log per epoch (first N)")
    g.add_argument("--log-all-samples", action="store_true", default=False,
                   help="Log ALL generated samples per epoch (overrides --log-samples)")
    g.add_argument("--no-run-log", action="store_true", default=False,
                   help="Disable run logging entirely")

    g.add_argument("--wait-pid", type=int, default=None,
                   help="Wait for this PID to exit before starting training "
                   "(useful for queueing CUDA jobs)")

    g = p.add_argument_group("Topology")
    g.add_argument("--topo", action="store_true", default=False,
                   help="Enable live topological barcode visualization")
    g.add_argument("--kelp-every", type=int, default=25,
                   help="Update topological kelp every N batches")
    g.add_argument("--topo-every", type=int, default=50,
                   help="Update topological barcodes every N batches")
    g.add_argument("--topo-max-points", type=int, default=200,
                   help="Max points to subsample for persistence computation")

    return p.parse_args()


args = parse_args()

def _is_int_str(s: str) -> bool:
    """Return True if s can be parsed as an integer."""
    try:
        int(s.strip())
        return True
    except (ValueError, TypeError, AttributeError):
        return False

def _tk_report_callback_exception(self, exc_type, exc_value, exc_tb):
    if exc_type is KeyboardInterrupt:
        return  # silently ignore
    # Fall back to default for real errors
    sys.__stderr__.write(f"Exception in Tkinter callback\n")
    import traceback
    traceback.print_exception(exc_type, exc_value, exc_tb)

tkinter.Tk.report_callback_exception = _tk_report_callback_exception

_interrupt_count = 0

def _sigint_handler(signum, frame):
    global _interrupt_count
    _interrupt_count += 1
    if _interrupt_count == 1:
        console.print("\n[bold yellow]⏳ Will stop after this epoch (Ctrl+C again to stop now)...[/]")
    else:
        console.print("\n[bold red]Stopping immediately.[/]")
        raise KeyboardInterrupt

signal.signal(signal.SIGINT, _sigint_handler)

# ════════════════════════════════════════════════════════════════════════════
# NEW: PID waiter
# ════════════════════════════════════════════════════════════════════════════

def wait_for_pid(pid: int, poll_interval: float = 2.0):
    """Block until the given PID is no longer running."""
    console.print(
            Panel(
                f"[bold yellow]⏳ Waiting for PID {pid} to finish before starting training...[/]\n"
                f"[dim]Polling every {poll_interval}s[/]",
                border_style="yellow",
                )
            )
    while True:
        try:
            os.kill(pid, 0)  # signal 0: doesn't kill, just checks existence
        except OSError as e:
            import errno
            if e.errno == errno.ESRCH:
                break
            raise
        except PermissionError:
            # Process exists but we don't own it — still alive
            pass
        else:
            # No exception → process is alive
            pass
        time.sleep(poll_interval)

        # Double-check with /proc on Linux or `ps` as fallback
        if not _pid_alive(pid):
            break

    console.print(f"[bold green]✅ PID {pid} is done. Proceeding to training.[/]")


def _pid_alive(pid: int) -> bool:
    """Return True if pid is still in the process table."""
    # Fast path: Linux /proc
    if os.path.isdir(f"/proc/{pid}"):
        return True
    # Fallback: use `ps`
    try:
        result = subprocess.run(
                ["ps", "-p", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                )
        return result.returncode == 0
    except FileNotFoundError:
        # No `ps` available, rely on os.kill alone
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

# ── Suppress X11/XIM spam ──────────────────────────────────────────────────
# The "key event is already fabricated" messages come from libX11.
# Redirect C-level stderr temporarily during matplotlib import.
_stderr_fd = _os.dup(2)
_devnull = _os.open(_os.devnull, _os.O_WRONLY)


def _suppress_c_stderr():
    _os.dup2(_devnull, 2)


def _restore_c_stderr():
    _os.dup2(_stderr_fd, 2)


console = Console()


# ════════════════════════════════════════════════════════════════════════════
# 1.  TOKENIZER  (HuggingFace-compatible save/load)
# ════════════════════════════════════════════════════════════════════════════

class BPETokenizer:
    """BPE tokenizer with the same interface as BPETokenizer."""

    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3}

    def __init__(self, hf_tokenizer: HFTokenizer = None):
        self._tok = hf_tokenizer
        # Cache special token IDs after training
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None
        self._sep_id = None
        if hf_tokenizer is not None:
            self._cache_special_ids()

    def _cache_special_ids(self):
        self._pad_id = self._tok.token_to_id("<pad>")
        self._bos_id = self._tok.token_to_id("<bos>")
        self._eos_id = self._tok.token_to_id("<eos>")
        self._sep_id = self._tok.token_to_id("<sep>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._pad_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def __call__(self, text: str, return_tensors: str = None, **kwargs) -> dict:
        ids = [self._bos_id] + self.encode(text) + [self._eos_id]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._tok.save(os.path.join(path, "tokenizer.json"))
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "BPETokenizer"}, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        hf_tok = HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(hf_tokenizer=hf_tok)

# ════════════════════════════════════════════════════════════════════════════
# 2.  DATA GENERATION (per-batch, on the fly)
# ════════════════════════════════════════════════════════════════════════════

def generate_single_sample(
        tokenizer: BPETokenizer,
        max_params: int = 4,
        max_ops: int = 6,
        allowed_ops: Optional[List[str]] = None,
        param_range: Tuple[int, int] = (-50, 50),
        max_seq_len: int = 2048,
        ) -> Optional[List[int]]:
    if allowed_ops is None:
        allowed_ops = ["add", "sub", "mul"]

    num_params = random.randint(2, max(2, max_params))
    num_ops = random.randint(1, max(1, max_ops))
    params = [random.randint(*param_range) for _ in range(num_params)]

    try:
        ir_code, result = generate_random_function(
                num_params=num_params,
                params=params,
                allowed_ops=allowed_ops,
                num_operations=num_ops,
                func_name="f",
                )
    except Exception:
        return None

    params_str = ",".join(str(p) for p in params)
    result_str = str(result)
    text = f"{ir_code}<sep>{params_str}<sep>{result_str}"

    ids = (
            [tokenizer.bos_token_id]
            + tokenizer.encode(text)
            + [tokenizer.eos_token_id]
            )

    prompt_part = f"{ir_code}<sep>{params_str}<sep>"
    prompt_len = 1 + len(tokenizer.encode(prompt_part))  # +1 for <bos>

    return ids, prompt_len

def generate_batch(
        tokenizer: BPETokenizer,
        batch_size: int,
        max_params: int = 3,
        max_ops: int = 4,
        allowed_ops: Optional[List[str]] = None,
        param_range: Tuple[int, int] = (-20, 20),
        max_seq_len: int = 2048,
        ) -> List[List[int]]:
    samples = []
    attempts = 0
    while len(samples) < batch_size and attempts < batch_size * 5:
        attempts += 1
        s = generate_single_sample(
                tokenizer, max_params, max_ops, allowed_ops, param_range, max_seq_len
                )
        if s is not None:
            samples.append(s)
    return samples


def collate_batch(batch, pad_id=0, tokenizer=None):
    """
    Collate a batch of (token_ids, prompt_len) pairs into padded tensors.

    Returns:
        input_ids:       (B, T)  — input tokens (shifted right)
        target_ids:      (B, T)  — target tokens (shifted left, prompt masked to pad_id)
        value_positions: (B,)    — index of the last <sep> in input_ids (for value head)
                                   Set to 0 ONLY when no <sep> tokens exist at all.
                                   Kept nonzero for unparseable answers so the structure
                                   penalty can still operate on the answer region.
        value_targets:   (B,)    — float tensor of expected integer results
                                   Set to NaN for unparseable answers.
        answer_parseable:(B,)    — bool tensor: True if the answer was a valid integer.
    """
    max_len = max(len(s) for s, _ in batch)
    input_ids, target_ids = [], []
    value_positions = []
    value_targets = []
    answer_parseable = []

    sep_id = None
    eos_id = None
    if tokenizer is not None:
        sep_id = tokenizer._tok.token_to_id("<sep>")
        eos_id = tokenizer.eos_token_id

    for s, prompt_len in batch:
        padded = s + [pad_id] * (max_len - len(s))
        inp = padded[:-1]
        tgt = padded[1:]

        # Mask out everything before the answer
        for i in range(min(prompt_len - 1, len(tgt))):
            tgt[i] = pad_id  # ignored by cross_entropy with ignore_index=0

        input_ids.append(inp)
        target_ids.append(tgt)

        # ── Find value_position: last <sep> in inp ──────────────────────
        vp = 0
        vt = float("nan")
        parseable = False

        if sep_id is not None:
            sep_positions = [i for i, tid in enumerate(inp) if tid == sep_id]
            if len(sep_positions) >= 2:
                vp = sep_positions[-1]  # last <sep> in input

                # Extract the answer tokens between last <sep> and <eos>/pad
                answer_start = sep_positions[-1] + 1
                answer_end = len(s) - 1  # exclude <eos> from original s
                answer_ids = s[answer_start:answer_end]

                try:
                    answer_str = tokenizer.decode(answer_ids).strip()
                    vt = float(int(answer_str))
                    parseable = True
                except (ValueError, TypeError):
                    # Answer is not a valid integer.
                    # KEEP vp nonzero — the answer region still exists,
                    # we just don't know the numeric target.
                    # vt stays NaN, parseable stays False.
                    pass

        value_positions.append(vp)
        value_targets.append(vt)
        answer_parseable.append(parseable)

    return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
            torch.tensor(value_positions, dtype=torch.long),
            torch.tensor(value_targets, dtype=torch.float),
            torch.tensor(answer_parseable, dtype=torch.bool),
            )

def build_tokenizer_from_samples(n_programs=1000, allowed_ops=None,
                                 max_params=4, max_ops=6,
                                 param_range=(-50, 50),
                                 bpe_vocab_size=512) -> BPETokenizer:
    """
    Generate n_programs random LLVM IR functions to build a tokenizer.
    """
    if allowed_ops is None:
        allowed_ops = ["add", "sub", "mul"]

    # ── Step 1: Generate corpus ─────────────────────────────────────────
    corpus: List[str] = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Generating LLVM IR corpus for tokenizer..."),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("• [cyan]{task.fields[status]}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            ) as progress:
        task = progress.add_task(
                "gen_corpus", total=n_programs, status="starting..."
                )
        success = 0
        fail = 0

        for i in range(n_programs):
            num_p = random.randint(2, max(2, max_params))
            num_o = random.randint(1, max(1, max_ops))
            params = [random.randint(*param_range) for _ in range(num_p)]
            try:
                ir_code, result = generate_random_function(
                        num_params=num_p, params=params,
                        allowed_ops=allowed_ops, num_operations=num_o,
                        func_name="f",
                        )
                text = f"{ir_code}<sep>{','.join(str(p) for p in params)}<sep>{result}"
                corpus.append(text)
                success += 1
            except Exception:
                fail += 1

            progress.update(
                    task, advance=1,
                    status=f"{success} ok, {fail} failed"
                    )

    console.print(
            f"  [green]✓[/] Corpus ready: [bold]{len(corpus)}[/] programs "
            f"([dim]{fail} generation failures skipped[/])"
            )

    # ── Step 2: Train BPE tokenizer ─────────────────────────────────────
    console.print(
            f"  [cyan]Training BPE tokenizer "
            f"(vocab_size={bpe_vocab_size}, corpus={len(corpus)} programs)...[/]"
            )

    hf_tokenizer = HFTokenizer(models.BPE())
    hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf_tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<pad>", "<bos>", "<eos>", "<sep>"]

    trainer_obj = trainers.BpeTrainer(
            vocab_size=bpe_vocab_size,
            special_tokens=special_tokens,
            min_frequency=2,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            )

    with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]Training BPE merges..."),
            TextColumn("(this may take a moment)"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            ) as progress:
        task = progress.add_task("bpe_train", total=None)  # indeterminate
        hf_tokenizer.train_from_iterator(corpus, trainer=trainer_obj)
        progress.update(task, completed=True)

    tokenizer = BPETokenizer(hf_tokenizer=hf_tokenizer)

    console.print(
            f"  [green]✓[/] BPE tokenizer trained — "
            f"vocab_size=[bold]{tokenizer.vocab_size}[/]"
            )

    # Show a sample encoding
    if corpus:
        sample = corpus[0][:80]
        encoded = tokenizer.encode(sample)
        console.print(
                f"  [dim]Sample: \"{sample}...\"[/]\n"
                f"  [dim]→ {len(encoded)} tokens (vs {len(sample)} chars)[/]"
                )

    return tokenizer

# ════════════════════════════════════════════════════════════════════════════
# 3.  TINY GPT MODEL  (HuggingFace-compatible save/load)
# ════════════════════════════════════════════════════════════════════════════

class LLVMGPTConfig:
    model_type = "llvm_gpt"

    def __init__(
            self,
            vocab_size: int = 90,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 4,
            max_seq_len: int = 2048,
            dropout: float = 0.1,
            **kwargs,
            ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str) -> "LLVMGPTConfig":
        with open(os.path.join(path, "config.json"), "r") as f:
            data = json.load(f)
        return cls(**data)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer(
                "mask",
                torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            scale = math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, T, C)
            return self.proj_drop(self.proj(out))
        except torch.cuda.OutOfMemoryError:
            console.print(f"[bold red]❌ Memory error. This can be caused by having a too large batch size or too little parameters.[/]")
            sys.exit(1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: LLVMGPTConfig):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
                *[
                    TransformerBlock(config.d_model, config.n_heads, config.max_seq_len, config.dropout)
                    for _ in range(config.n_layers)
                    ]
                )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        # ── Value regression head ───────────────────────────────────────
        # Projects the last hidden state at a chosen position to a scalar
        # predicted integer value. This is fully differentiable.
        self.value_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
                )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            value_positions: Optional[torch.Tensor] = None,
            output_hidden_states: bool = False,
            **kwargs,
            ):
        """
        Args:
            input_ids:       (B, T) token indices
            labels:          (B, T) target token indices (for CE loss)
            attention_mask:  unused, kept for API compat
            value_positions: (B,) int tensor — index into the sequence at which
                             to read the hidden state for the value head.
                             Typically the position of the last <sep> token
                             (i.e. the position just before the answer starts).
                             If None, value_preds will be None.
            output_hidden_states: if True, return all intermediate hidden states
        """
        idx = input_ids
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        hidden_states_list = [x] if output_hidden_states else None
        for blk in self.blocks:
            x = blk(x)
            if output_hidden_states:
                hidden_states_list.append(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # ── Cross-entropy loss (token-level) ────────────────────────────
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0,
                    )

        # ── Value head prediction ───────────────────────────────────────
        # Gather the hidden state at the specified position for each sample
        value_preds = None
        if value_positions is not None:
            # value_positions: (B,) — index into T dimension
            # x shape: (B, T, D)
            # We want x[b, value_positions[b], :] for each b
            gather_idx = value_positions.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            gather_idx = gather_idx.expand(-1, -1, x.shape[-1])       # (B, 1, D)
            pooled = x.gather(1, gather_idx).squeeze(1)               # (B, D)
            value_preds = self.value_head(pooled).squeeze(-1)         # (B,)

        return _ModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=tuple(hidden_states_list) if output_hidden_states else None,
                last_hidden_state=x,
                value_preds=value_preds,
                )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TinyGPT":
        config = LLVMGPTConfig.from_pretrained(path)
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        model = cls(config)
        state_dict = torch.load(
                os.path.join(path, "pytorch_model.bin"),
                map_location="cpu",
                weights_only=True,
                )
        model.load_state_dict(state_dict)
        return model

def compute_structured_loss(
        model: 'TinyGPT',
        inp: torch.Tensor,
        tgt: torch.Tensor,
        value_positions: torch.Tensor,
        value_targets: torch.Tensor,
        answer_parseable: torch.Tensor,
        tokenizer: 'BPETokenizer',
        device: str,
        alpha_value: float = 0.1,
        alpha_structure: float = 0.5,
        alpha_length: float = 0.3,
        use_log_scale: bool = True,
        ) -> Tuple[torch.Tensor, float, float, float, float]:
    """
    Combined loss:
      1. Cross-entropy (teacher-forced, standard)
      2. Value regression (L1 on value head)
      3. Generation reward loss — ACTUALLY GENERATE, score the output,
         and use the log-probability of the generated sequence weighted
         by the reward as a REINFORCE-style gradient signal.
      4. Length penalty — penalizes the model when the predicted answer
         length (measured via EOS token probability) differs from the
         expected answer length. This is fully differentiable.
    """
    output = model(
            input_ids=inp,
            labels=tgt,
            value_positions=value_positions,
            )

    ce_loss = output.loss
    logits = output.logits
    value_preds = output.value_preds

    # ── Value regression loss ───────────────────────────────────────────
    val_loss_val = 0.0

    # ── Sign-aware value regression loss ────────────────────────────────
    value_loss_term = torch.tensor(0.0, device=device)
    if value_preds is not None and alpha_value > 0:
        valid_mask = (value_positions > 0) & answer_parseable & (~torch.isnan(value_targets))
        if valid_mask.any():
            vp = value_preds[valid_mask]
            vt = value_targets[valid_mask]

            # Component 1: Sign correctness
            # If signs match → 0 penalty; if signs differ → penalty
            # Uses a soft sign via tanh so gradients flow smoothly
            pred_sign = torch.tanh(vp * 0.5)       # soft sign: ~-1 or ~+1
            target_sign = torch.sign(vt)            # hard sign: -1, 0, or +1
            # When signs match: (1 - pred_sign * target_sign) ≈ 0
            # When signs differ: (1 - pred_sign * target_sign) ≈ 2
            sign_penalty = (1.0 - pred_sign * target_sign).mean()

            # Component 2: Log-scaled magnitude difference
            # This compresses large differences so they don't dominate,
            # but still distinguishes "off by 1" from "off by 1000"
            if use_log_scale:
                magnitude_loss = torch.log1p(torch.abs(vp - vt)).mean()
            else:
                magnitude_loss = F.l1_loss(vp, vt)

            # Component 3: Relative error (treats large vs small numbers fairly)
            # Being off by 5 when the answer is 10 is worse than
            # being off by 5 when the answer is 1000
            relative_error = (torch.abs(vp - vt) / (torch.abs(vt) + 1.0)).mean()

            # Combine: sign matters a LOT, magnitude matters proportionally
            value_loss_term = (
                    0.4 * sign_penalty +
                    0.35 * magnitude_loss +
                    0.25 * relative_error
                    )
            val_loss_val = value_loss_term.item()

    # ── Length penalty loss (differentiable) ────────────────────────────
    length_loss_term = _compute_length_penalty(
            logits=logits,
            tgt=tgt,
            value_positions=value_positions,
            answer_parseable=answer_parseable,
            tokenizer=tokenizer,
            device=device,
            )

    # ── Generation reward loss (REINFORCE-style) ────────────────────────
    gen_loss_term, parsability_rate = _compute_generation_reward_loss(
            model=model,
            inp=inp,
            value_positions=value_positions,
            value_targets=value_targets,
            answer_parseable=answer_parseable,
            tokenizer=tokenizer,
            device=device,
            max_gen_len=20,
            )

    structure_loss_val = gen_loss_term.item() if isinstance(gen_loss_term, torch.Tensor) else 0.0

    # ── Combine ─────────────────────────────────────────────────────────
    total_loss = (
            ce_loss
            + alpha_value * value_loss_term
            + alpha_structure * gen_loss_term
            + alpha_length * length_loss_term
            )

    return (
            total_loss,
            ce_loss.item(),
            val_loss_val,
            structure_loss_val,
            parsability_rate,
            )


def _compute_length_penalty(
        logits: torch.Tensor,           # (B, T, V)
        tgt: torch.Tensor,              # (B, T)
        value_positions: torch.Tensor,  # (B,)
        answer_parseable: torch.Tensor, # (B,)
        tokenizer: 'BPETokenizer',
        device: str,
        ) -> torch.Tensor:
    """
    Differentiable length penalty.

    For each sample, we know the expected answer length (number of tokens
    between the last <sep> and <eos>/<pad> in the target). We compute the
    model's "predicted length" as the expected position of EOS under the
    model's softmax distribution over the answer region:

        predicted_len = sum_{t in answer_region} t * P(EOS at position t)

    where P(EOS at position t) is the softmax probability of the EOS token
    at each answer position, normalized to form a distribution.

    The loss is the squared difference between expected and predicted lengths,
    scaled by 1/expected_len to be scale-invariant (so a 1-token error on a
    2-token answer is penalized the same as a 5-token error on a 10-token answer).

    This is fully differentiable because it operates on softmax(logits).
    """
    B, T, V = logits.shape
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    if eos_id is None:
        return torch.tensor(0.0, device=device)

    probs = F.softmax(logits, dim=-1)  # (B, T, V)

    length_losses = []

    for b in range(B):
        vp = value_positions[b].item()
        if vp == 0:
            continue

        # Find expected answer length from target
        answer_start = vp  # position after last <sep> in the shifted target
        answer_end = T
        for t_idx in range(answer_start, T):
            tok = tgt[b, t_idx].item()
            if tok == pad_id or tok == eos_id:
                answer_end = t_idx
                break

        expected_len = answer_end - answer_start
        if expected_len <= 0:
            continue

        # The full answer region including the EOS position
        # We look at answer_start to answer_end (inclusive of EOS position)
        region_end = min(answer_end + 1, T)  # +1 to include the EOS slot
        region_len = region_end - answer_start
        if region_len <= 0:
            continue

        # P(EOS) at each position in the answer region
        eos_probs = probs[b, answer_start:region_end, eos_id]  # (region_len,)

        # Normalize to form a distribution (where does the model think EOS is?)
        eos_dist = eos_probs / (eos_probs.sum() + 1e-8)

        # Expected position of EOS (0-indexed from answer_start)
        positions = torch.arange(region_len, dtype=torch.float, device=device)
        predicted_len = (eos_dist * positions).sum()

        # Scale-invariant squared error
        # Dividing by expected_len makes a 1-token error on a 2-token answer
        # equivalent to a 5-token error on a 10-token answer
        len_diff = predicted_len - float(expected_len)
        scale = max(float(expected_len), 1.0)
        length_loss = (len_diff / scale) ** 2

        length_losses.append(length_loss)

    if not length_losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(length_losses).mean()

def _compute_generation_reward_loss(
        model: 'TinyGPT',
        inp: torch.Tensor,
        value_positions: torch.Tensor,
        value_targets: torch.Tensor,
        answer_parseable: torch.Tensor,
        tokenizer: 'BPETokenizer',
        device: str,
        max_gen_len: int = 20,
        ) -> Tuple[torch.Tensor, float]:
    """
    Generate from the model, score using _compute_structure_penalty,
    and use the score as a REINFORCE reward signal.

    The structure penalty (0=perfect, 6=garbage) is converted to a
    reward in [-1, +1] via:  reward = 1.0 - (penalty / 3.0)
    clamped to [-1, +1].

    This means:
      penalty=0.0 (perfect)     → reward = +1.0
      penalty=0.5 (close int)   → reward = +0.83
      penalty=1.0 (padded int)  → reward = +0.67
      penalty=3.0 (partial num) → reward =  0.0  (baseline)
      penalty=5.0 (garbage)     → reward = -0.67
      penalty=6.0 (empty)       → reward = -1.0
    """
    B = inp.shape[0]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    all_rewards = []
    all_log_probs = []
    n_parseable = 0
    n_with_region = 0

    for b in range(B):
        vp = value_positions[b].item()
        if vp == 0:
            continue
        n_with_region += 1

        vt = value_targets[b].item()
        is_parseable = answer_parseable[b].item()

        # ── Get expected string and int ─────────────────────────────
        if is_parseable and not math.isnan(vt):
            expected_int = int(vt)
            expected_str = str(expected_int)
        else:
            expected_int = 0
            expected_str = "0"

        # ── Prompt = everything up to and including last <sep> ──────
        prompt = inp[b, :vp + 1].tolist()

        # ── Generate greedily, collecting log-probs WITH gradient ───
        generated_ids = []
        log_prob_sum = torch.tensor(0.0, device=device)
        current_ids = list(prompt)

        for step in range(max_gen_len):
            input_tensor = torch.tensor(
                    [current_ids[-model.max_seq_len:]],
                    dtype=torch.long, device=device,
                    )
            out = model(input_ids=input_tensor)
            step_logits = out.logits[0, -1, :]
            step_log_probs = F.log_softmax(step_logits, dim=-1)

            next_token = step_logits.argmax().item()
            log_prob_sum = log_prob_sum + step_log_probs[next_token]

            if next_token == eos_id:
                break
            generated_ids.append(next_token)
            current_ids.append(next_token)

        # ── Decode generated output ─────────────────────────────────
        if generated_ids:
            gen_str = tokenizer.decode(generated_ids).strip()
        else:
            gen_str = ""

        # ── Score using your existing penalty function ──────────────
        penalty = _compute_structure_penalty(
                expected_str=expected_str,
                expected_int=expected_int,
                predicted_str=gen_str,
                )

        # ── Convert penalty → reward ────────────────────────────────
        # penalty=0 → reward=+1, penalty=3 → reward=0, penalty=6 → reward=-1
        reward = 1.0 - (penalty / 3.0)
        reward = max(-1.0, min(1.0, reward))

        # Track parsability
        pred_cleaned = ''.join(
                c for c in gen_str if c.isascii() and c.isprintable()
                ).strip()
        try:
            int(pred_cleaned)
            n_parseable += 1
        except (ValueError, TypeError):
            pass

        all_rewards.append(reward)
        all_log_probs.append(log_prob_sum)

    # ── REINFORCE loss ──────────────────────────────────────────────────
    if not all_log_probs:
        return torch.tensor(0.0, device=device), 0.0

    parsability_rate = n_parseable / max(n_with_region, 1)

    # Baseline: mean reward (variance reduction)
    mean_reward = sum(all_rewards) / len(all_rewards)

    # Policy gradient: -E[(reward - baseline) * log_prob]
    loss_terms = []
    for reward, lp in zip(all_rewards, all_log_probs):
        advantage = reward - mean_reward
        loss_terms.append(-advantage * lp)

    if loss_terms:
        gen_loss = torch.stack(loss_terms).mean()
    else:
        gen_loss = torch.tensor(0.0, device=device)

    return gen_loss, parsability_rate

def _build_digit_token_set(tokenizer: 'BPETokenizer') -> torch.Tensor:
    """
    Return a boolean tensor of shape (vocab_size,) that is True for tokens
    that represent digits, minus signs, or digit-containing subwords.

    This is computed once and cached on the tokenizer object.
    """
    if hasattr(tokenizer, '_digit_mask_cache'):
        return tokenizer._digit_mask_cache

    vocab_size = tokenizer.vocab_size
    mask = torch.zeros(vocab_size, dtype=torch.bool)

    digit_chars = set('0123456789-')

    for token_id in range(vocab_size):
        try:
            token_str = tokenizer.decode([token_id])
            # A token is "numeric" if ALL its printable characters are digits or minus
            printable = ''.join(c for c in token_str if c.isascii() and c.isprintable()).strip()
            if printable and all(c in digit_chars for c in printable):
                mask[token_id] = True
        except Exception:
            continue

    tokenizer._digit_mask_cache = mask
    return mask

def _compute_differentiable_structure_penalty(
        logits: torch.Tensor,           # (B, T, V)
        tgt: torch.Tensor,              # (B, T)
        value_positions: torch.Tensor,  # (B,)
        value_targets: torch.Tensor,    # (B,)
        answer_parseable: torch.Tensor, # (B,)
        tokenizer: 'BPETokenizer',
        device: str,
        temperature: float = 1.0,
        ) -> Tuple[torch.Tensor, float]:
    """
    Compute a fully differentiable structure penalty over the answer region.

    This operates entirely in logit/probability space, so gradients flow
    back through the model's parameters via the logits.

    KEY FIX: For samples where the answer region exists (value_positions > 0)
    but the target value is unparseable (answer_parseable=False), we STILL
    apply the digit-focus and length-calibration penalties. This ensures
    the model always receives gradient signal to produce numeric tokens in
    the answer region, even when we can't compute the KL target.

    For parseable samples, all three sub-penalties apply:
      1. Digit focus loss
      2. Soft target KL divergence
      3. Length calibration loss

    For unparseable samples (answer region exists but value is NaN):
      1. Digit focus loss (FULL WEIGHT — this is the primary learning signal)
      2. Soft target KL: SKIPPED (we don't know the correct tokens)
      3. Length calibration loss (still applies — answer should end with EOS)
      4. Anti-garbage penalty: extra penalty proportional to probability mass
         on non-digit tokens, with a higher weight than for parseable samples.

    Returns:
        (loss_tensor, loss_scalar_for_logging)
    """
    B, T, V = logits.shape
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # Get digit token mask (cached after first call)
    digit_mask = _build_digit_token_set(tokenizer).to(device)  # (V,)

    # Log-probabilities for the full sequence
    log_probs = F.log_softmax(logits / temperature, dim=-1)  # (B, T, V)
    probs = log_probs.exp()  # (B, T, V)

    # Accumulators for the sub-penalties
    digit_focus_losses = []
    kl_losses = []
    length_losses = []
    anti_garbage_losses = []

    # Counters for diagnostics
    n_parseable = 0
    n_unparseable_with_region = 0
    n_skipped = 0

    # Soft target: 85% on correct token, 15% spread over other digit tokens
    CORRECT_MASS = 0.85
    DIGIT_SPREAD_MASS = 0.15

    for b in range(B):
        vp_idx = value_positions[b].item()
        if vp_idx == 0:
            # No answer region at all (no <sep> tokens found)
            n_skipped += 1
            continue

        is_parseable = answer_parseable[b].item()
        vt = value_targets[b].item()

        # Determine the answer region in the target
        answer_start = vp_idx
        answer_end = T

        # Find where the answer ends (first pad or eos in target)
        for t_idx in range(answer_start, T):
            tok = tgt[b, t_idx].item()
            if tok == pad_id or tok == eos_id:
                answer_end = t_idx
                break

        if answer_end <= answer_start:
            n_skipped += 1
            continue

        answer_len = answer_end - answer_start

        # ── Sub-penalty 1: Digit focus (ALWAYS applied) ────────────────
        # For each answer position, compute P(any digit token) and penalize
        # 1 - P(digit). Applied to ALL samples with an answer region.
        answer_probs = probs[b, answer_start:answer_end, :]  # (answer_len, V)
        digit_prob_mass = answer_probs[:, digit_mask].sum(dim=-1)  # (answer_len,)
        digit_focus_loss = (1.0 - digit_prob_mass).mean()
        digit_focus_losses.append(digit_focus_loss)

        # ── Sub-penalty 3: Length calibration (ALWAYS applied) ──────────
        if eos_id is not None and answer_len > 1:
            early_eos_probs = probs[b, answer_start:answer_end - 1, eos_id]
            early_eos_penalty = early_eos_probs.mean()

            if answer_end < T:
                eos_at_end = probs[b, answer_end, eos_id]
                late_eos_penalty = (1.0 - eos_at_end)
            else:
                late_eos_penalty = torch.tensor(0.0, device=device)

            length_loss = 0.5 * early_eos_penalty + 0.5 * late_eos_penalty
            length_losses.append(length_loss)

        if is_parseable and not math.isnan(vt):
            # ── Sub-penalty 2: Soft target KL (parseable only) ──────────
            n_parseable += 1

            answer_log_probs = log_probs[b, answer_start:answer_end, :]
            answer_targets_tok = tgt[b, answer_start:answer_end]

            n_digit_tokens = digit_mask.sum().float().clamp(min=1.0)
            soft_targets = torch.zeros(answer_len, V, device=device)
            soft_targets[:, digit_mask] = DIGIT_SPREAD_MASS / n_digit_tokens

            for pos_idx in range(answer_len):
                target_tok = answer_targets_tok[pos_idx].item()
                if target_tok != pad_id and target_tok != eos_id:
                    soft_targets[pos_idx, target_tok] += CORRECT_MASS
                    row_sum = soft_targets[pos_idx].sum()
                    if row_sum > 0:
                        soft_targets[pos_idx] = soft_targets[pos_idx] / row_sum

            kl = F.kl_div(
                    answer_log_probs,
                    soft_targets,
                    reduction='batchmean',
                    log_target=False,
                    )
            kl_losses.append(kl)

        else:
            # ── Unparseable sample with answer region ───────────────────
            # We can't compute KL (don't know the target), but we CAN
            # apply an EXTRA anti-garbage penalty: penalize probability
            # mass on non-digit, non-eos, non-pad tokens in the answer
            # region. This is strictly stronger than digit_focus alone.
            n_unparseable_with_region += 1

            # Compute entropy of the answer region — high entropy means
            # the model is uncertain, which is better than confidently
            # producing garbage. Penalize LOW entropy on non-digit tokens.
            non_digit_mask = ~digit_mask.clone()
            # Don't penalize eos/pad (they're structural, not garbage)
            if eos_id is not None and eos_id < V:
                non_digit_mask[eos_id] = False
            if pad_id is not None and pad_id < V:
                non_digit_mask[pad_id] = False

            answer_probs_b = probs[b, answer_start:answer_end, :]
            # Total probability on garbage (non-digit, non-structural) tokens
            garbage_prob_mass = answer_probs_b[:, non_digit_mask].sum(dim=-1)
            # Penalty: mean garbage probability across answer positions
            # This is differentiable and pushes mass away from garbage tokens
            anti_garbage = garbage_prob_mass.mean()
            anti_garbage_losses.append(anti_garbage)

    # ── Aggregate ───────────────────────────────────────────────────────
    zero = torch.tensor(0.0, device=device, requires_grad=False)

    if digit_focus_losses:
        avg_digit = torch.stack(digit_focus_losses).mean()
    else:
        avg_digit = zero

    if kl_losses:
        avg_kl = torch.stack(kl_losses).mean()
    else:
        avg_kl = zero

    if length_losses:
        avg_length = torch.stack(length_losses).mean()
    else:
        avg_length = zero

    if anti_garbage_losses:
        avg_anti_garbage = torch.stack(anti_garbage_losses).mean()
    else:
        avg_anti_garbage = zero

    # ── Adaptive weighting ──────────────────────────────────────────────
    # When most samples are unparseable, the digit focus and anti-garbage
    # penalties should dominate. When most are parseable, KL dominates.
    total_with_region = n_parseable + n_unparseable_with_region
    if total_with_region > 0:
        parseable_ratio = n_parseable / total_with_region
    else:
        parseable_ratio = 0.0

    # Base weights (when all samples are parseable)
    WEIGHT_DIGIT = 0.2
    WEIGHT_KL = 0.6
    WEIGHT_LENGTH = 0.2
    WEIGHT_ANTI_GARBAGE = 0.0

    # Shift weights when samples are unparseable:
    # - KL weight scales down (can't compute it for unparseable)
    # - Digit focus and anti-garbage scale up to compensate
    effective_kl_weight = WEIGHT_KL * parseable_ratio
    unparseable_ratio = 1.0 - parseable_ratio
    effective_digit_weight = WEIGHT_DIGIT + 0.3 * unparseable_ratio
    effective_garbage_weight = 0.4 * unparseable_ratio
    effective_length_weight = WEIGHT_LENGTH

    # Normalize so weights sum to 1.0 (preserves loss scale)
    total_weight = (effective_digit_weight + effective_kl_weight +
                    effective_length_weight + effective_garbage_weight)
    if total_weight > 0:
        effective_digit_weight /= total_weight
        effective_kl_weight /= total_weight
        effective_length_weight /= total_weight
        effective_garbage_weight /= total_weight

    total_structure = (
            effective_digit_weight * avg_digit
            + effective_kl_weight * avg_kl
            + effective_length_weight * avg_length
            + effective_garbage_weight * avg_anti_garbage
            )

    # Ensure we always return a tensor that participates in the graph
    # even if all sub-losses are zero (prevents silent gradient death)
    if not total_structure.requires_grad and total_with_region > 0:
        # This shouldn't happen if any sub-loss was computed from probs,
        # but as a safety net: add a tiny term from the logits directly
        # so the graph stays connected.
        total_structure = total_structure + 0.0 * logits.sum() * 0.0

    # ── Parsability penalty WITH gradients ──────────────────────────────
    # When samples are unparseable, penalize the model's confidence on
    # non-digit tokens in the answer region. This is DIFFERENTIABLE
    # because it operates on the logit probabilities directly.
    if total_with_region > 0 and n_parseable < total_with_region:
        unparseable_frac = 1.0 - (n_parseable / total_with_region)
        # Scale up the existing digit_focus and anti_garbage losses
        # for unparseable samples — these already have gradients
        boost = torch.tensor(unparseable_frac, device=device) 
        # Add a GRADIENT-CARRYING term: amplify avg_digit (which has grads)
        parsability_boost = boost * avg_digit * 2.0
        total_structure = total_structure + parsability_boost

    structure_scalar = total_structure.item() if isinstance(total_structure, torch.Tensor) else 0.0

    return total_structure, structure_scalar

def _compute_structure_penalty(
        expected_str: str,
        expected_int: int,
        predicted_str: str,
        ) -> float:
    """
    Compute a scalar penalty for a single prediction (NON-DIFFERENTIABLE).

    This is retained for logging, visualization, and the prediction diff
    plot — but is NO LONGER used in the loss computation.

    Hierarchy (strictly ordered, NO overlap between levels):
      0.0            — perfect integer match (numeric comparison)
      (0, 0.99]      — wrong integer, valid format
                       Sub-scoring accounts for:
                         • Sign correctness (same sign = lower penalty)
                         • Magnitude difference (log-scaled)
                         • Relative error (proportional to target size)
      [1.0, 1.49]    — integer with leading/trailing whitespace
      [1.5, 1.99]    — starts with a valid integer but has trailing chars
      [2.0, 2.99]    — float-like string
      [3.0, 4.99]    — partially numeric / dash-prefixed
      [5.0, 5.99]    — pure garbage
      6.0            — empty output
    """
    if not predicted_str:
        return 6.0

    # Clean BPE artifacts
    cleaned = ''.join(c for c in predicted_str if c.isascii() and c.isprintable())
    if not cleaned:
        return 6.0

    stripped = cleaned.strip()

    # ── Helper: sign-aware, magnitude-sensitive sub-score ───────────────
    # Returns a value in [0.0, 1.0) where:
    #   0.0 = perfect match
    #   Higher = worse
    # Accounts for:
    #   1. Sign match/mismatch (large bonus/penalty)
    #   2. Log-scaled absolute difference (compresses large diffs)
    #   3. Relative error (off-by-5 on target=10 is worse than on target=1000)
    def _sub_score(expected: float, predicted: float) -> float:
        diff = abs(expected - predicted)
        if diff == 0:
            return 0.0

        # Component 1: Sign match
        # Same sign (or one is zero) → 0.0 penalty
        # Opposite signs → 0.35 penalty
        exp_sign = (expected > 0) - (expected < 0)   # -1, 0, or +1
        pred_sign = (predicted > 0) - (predicted < 0)
        if exp_sign == 0 or pred_sign == 0:
            sign_penalty = 0.0
        elif exp_sign == pred_sign:
            sign_penalty = 0.0
        else:
            sign_penalty = 0.35

        # Component 2: Log-scaled magnitude difference
        # log1p compresses large diffs: off-by-1 → 0.69, off-by-1000 → 6.9
        # Normalize with a sigmoid-like curve to [0, 1)
        log_diff = math.log1p(diff)
        magnitude_score = 1.0 - 1.0 / (1.0 + log_diff * 0.3)
        # Scale to contribute at most 0.40
        magnitude_component = 0.40 * magnitude_score

        # Component 3: Relative error
        # off-by-5 when target=10 → rel=0.45, when target=1000 → rel=0.005
        # Clamp to [0, 1] and scale to contribute at most 0.24
        rel_error = diff / (abs(expected) + 1.0)
        rel_component = 0.24 * min(1.0, rel_error / (1.0 + rel_error))

        # Combine (max possible ≈ 0.35 + 0.40 + 0.24 = 0.99)
        return min(0.99, sign_penalty + magnitude_component + rel_component)

    # ── Level 0: Perfect match or close integer ────────────────────────
    if _RE_INT.match(stripped):
        try:
            pred_int = int(stripped)
            diff = abs(expected_int - pred_int)
            if diff == 0:
                return 0.0
            return _sub_score(expected_int, pred_int)
        except (ValueError, OverflowError):
            pass

    # ── Level 1: Whitespace-padded integer ─────────────────────────────
    padded_match = _RE_PADDED_INT.match(cleaned)
    if padded_match:
        try:
            pred_int = int(padded_match.group(1))
            sub = _sub_score(expected_int, pred_int)
            return 1.0 + 0.49 * sub
        except (ValueError, OverflowError):
            pass

    # ── Level 1.5: Starts with valid integer + trailing stuff ──────────
    leading_match = _RE_LEADING_INT.match(cleaned)
    if leading_match:
        int_part = leading_match.group(1).strip()
        trailing = leading_match.group(2)
        if int_part and trailing:
            try:
                pred_int = int(int_part)
                sub = _sub_score(expected_int, pred_int)
                base = 1.5 + 0.3 * sub
                trailing_penalty = min(0.19, len(trailing) * 0.03)
                return base + trailing_penalty
            except (ValueError, OverflowError):
                pass

    # ── Level 2: Float-like string ─────────────────────────────────────
    if _RE_FLOAT.match(stripped):
        try:
            pred_float = float(stripped)
            sub = _sub_score(expected_int, pred_float)
            return 2.0 + 0.99 * sub
        except (ValueError, OverflowError):
            return 2.5

    # ── Level 2.5: Float with trailing ─────────────────────────────────
    leading_float_match = _RE_LEADING_FLOAT.match(cleaned)
    if leading_float_match:
        float_part = leading_float_match.group(1).strip()
        trailing = leading_float_match.group(2)
        if float_part and trailing:
            try:
                pred_float = float(float_part)
                sub = _sub_score(expected_int, pred_float)
                return 2.5 + 0.49 * sub
            except (ValueError, OverflowError):
                pass

    # ── Shared length penalty ──────────────────────────────────────────
    len_diff = abs(len(cleaned) - len(expected_str))
    len_penalty = min(0.5, math.log1p(len_diff) * 0.2)

    # ── Level 3–4: Partially numeric ───────────────────────────────────
    if cleaned.startswith('-') or _RE_PARTIAL_NUMERIC.match(cleaned):
        numeric_chars = sum(1 for c in cleaned if c in '-0123456789.')
        numeric_ratio = numeric_chars / max(len(cleaned), 1)
        return 3.0 + (1.0 - numeric_ratio) * 1.5 + len_penalty

    # ── Level 5: Pure garbage ──────────────────────────────────────────
    return 5.0 + len_penalty + min(0.49, math.log1p(len(cleaned)) * 0.15)

class _ModelOutput(dict):
    def __init__(self, loss=None, logits=None, hidden_states=None,
                 last_hidden_state=None, value_preds=None):
        super().__init__(
                loss=loss,
                logits=logits,
                hidden_states=hidden_states,
                last_hidden_state=last_hidden_state,
                value_preds=value_preds,
                )
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state
        self.value_preds = value_preds

    def __getitem__(self, key):
        return getattr(self, key)

# ════════════════════════════════════════════════════════════════════════════
# 4.  ANALYTICAL MODEL CONFIG
# ════════════════════════════════════════════════════════════════════════════

def estimate_params(vocab_size: int, d_model: int, n_layers: int, max_seq_len: int) -> int:
    emb = vocab_size * d_model + max_seq_len * d_model
    blocks = n_layers * (12 * d_model ** 2 + 13 * d_model)
    ln_f = 2 * d_model
    return emb + blocks + ln_f


def find_model_config(
        vocab_size: int,
        target_params: int = 1_000,
        max_seq_len: int = 2048,
        ) -> dict:
    best = None
    best_diff = float("inf")
    for d_model in range(8, 512, 2):
        for n_heads in (1, 2, 4, 8):
            if d_model % n_heads != 0:
                continue
            for n_layers in range(1, 24):
                n = estimate_params(vocab_size, d_model, n_layers, max_seq_len)
                diff = abs(n - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best = {
                            "d_model": d_model,
                            "n_heads": n_heads,
                            "n_layers": n_layers,
                            "estimated_params": n,
                            }
    return best


# ════════════════════════════════════════════════════════════════════════════
# 5.  OPTIMIZER / SCHEDULER FACTORIES
# ════════════════════════════════════════════════════════════════════════════

def build_warmup_cosine(optimizer, epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════
# 6.  KEYBOARD LISTENER — Ctrl+Up / Ctrl+Down to adjust epochs
# ════════════════════════════════════════════════════════════════════════════

class EpochController:
    """
    Mutable epoch target. Listens for keyboard input in a background thread.

    Press  +  (or =)  →  add `step` epochs
    Press  -          →  remove `step` epochs (min: current epoch)
    Press  q          →  finish after current epoch
    """

    def __init__(self, initial_epochs: int, step: int = 10, plotter=None):
        self.total_epochs = initial_epochs
        self.step = step
        self._min_epoch = 1
        self._lock = __import__("threading").Lock()
        self._thread = None
        self._running = False
        self._quit = False
        self._plotter = plotter  # Reference to LivePlotter for reopen

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def set_min(self, current_epoch: int):
        with self._lock:
            self._min_epoch = current_epoch

    @property
    def epochs(self) -> int:
        with self._lock:
            return self.total_epochs

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self._quit

    def _listen(self):
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self._running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1)
                        if ch in ("+", "="):
                            with self._lock:
                                self.total_epochs += self.step
                            console.print(
                                    f"\n[bold green]  ↑ Epochs → "
                                    f"{self.total_epochs} (+{self.step})[/]\n"
                                    )
                        elif ch == "-":
                            with self._lock:
                                self.total_epochs = max(
                                        self._min_epoch, self.total_epochs - self.step
                                        )
                            console.print(
                                    f"\n[bold red]  ↓ Epochs → "
                                    f"{self.total_epochs} (-{self.step})[/]\n"
                                    )
                        elif ch == "r":
                            # Request plot window reopen
                            if self._plotter is not None:
                                self._plotter.request_reopen()
                                console.print(
                                        "\n[bold cyan]  📊 Reopen plot requested...[/]\n"
                                        )
                        elif ch == "q":
                            with self._lock:
                                self._quit = True
                                self.total_epochs = self._min_epoch
                            console.print(
                                    "\n[bold yellow]  ⏹ Finishing after current epoch...[/]\n"
                                    )
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

# ════════════════════════════════════════════════════════════════════════════
# 7.  LIVE PLOTTER
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# 7.  LIVE PLOTTER  (with integrated TDA barcode + birth/death panels)
# ════════════════════════════════════════════════════════════════════════════

try:
    from ripser import ripser as _ripser_fn
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False

try:
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

@torch.no_grad()
def get_batch_predictions(model, tokenizer, batch, device, max_gen_len=20):
    """
    Given a batch of (token_ids, prompt_len) pairs, generate predictions
    and compare to expected outputs.
    Returns: List of (expected_str, predicted_str, is_correct)
    """
    model.eval()
    predictions = []

    sep_id = tokenizer._tok.token_to_id("<sep>")

    for token_ids, prompt_len in batch[:8]:
        # Find the answer portion by locating <sep> tokens in the ID sequence
        # Format: <bos> [ir_code] <sep> [params] <sep> [result] <eos>
        # We need to find the second <sep> and extract the answer after it

        sep_positions = [i for i, tid in enumerate(token_ids) if tid == sep_id]
        if len(sep_positions) < 2:
            continue

        # The answer starts after the second <sep>
        answer_start = sep_positions[1] + 1
        # Find <eos> or end of sequence
        eos_id = tokenizer.eos_token_id
        answer_end = len(token_ids)
        for i in range(answer_start, len(token_ids)):
            if token_ids[i] == eos_id:
                answer_end = i
                break

        expected_ids = token_ids[answer_start:answer_end]
        expected_answer = tokenizer.decode(expected_ids).strip()

        # Use prompt_len to get the prompt (everything up to and including second <sep>)
        prompt_ids = token_ids[:prompt_len]

        # Greedy decode
        generated = list(prompt_ids)
        for _ in range(max_gen_len):
            inp = torch.tensor(
                    [generated[-model.max_seq_len:]],
                    dtype=torch.long, device=device,
                    )
            output = model(input_ids=inp)
            logits = output.logits[0, -1, :]
            next_token = logits.argmax().item()

            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)

        # Decode only the generated portion (after prompt)
        generated_ids = generated[prompt_len:]
        generated_answer = tokenizer.decode(generated_ids).strip()

        # Clean up any special token remnants
        for special in ["<eos>", "<pad>", "<bos>", "<sep>"]:
            generated_answer = generated_answer.replace(special, "")
        generated_answer = generated_answer.strip()

        is_correct = generated_answer.strip() == expected_answer.strip()
        predictions.append((expected_answer, generated_answer, is_correct))

    model.train()
    return predictions

def get_gpu_info() -> Optional[Dict[str, any]]:
    """
    Query nvidia-smi for GPU stats. Returns a dict with GPU info,
    or None if nvidia-smi is not available or fails.
    """
    try:
        result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    ],
                capture_output=True,
                text=True,
                timeout=5,
                )
        if result.returncode != 0:
            return None

        line = result.stdout.strip().split("\n")[0]  # First GPU
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return None

        info = {
                "gpu_name": parts[0],
                "gpu_mem_used_mb": int(float(parts[1])),
                "gpu_mem_total_mb": int(float(parts[2])),
                "gpu_mem_free_mb": int(float(parts[3])),
                "gpu_util_pct": int(float(parts[4])),
                "gpu_temp_c": int(float(parts[5])),
                }

        # power.draw can sometimes be "[N/A]" on some GPUs
        try:
            info["gpu_power_w"] = round(float(parts[6]), 1)
        except (ValueError, IndexError):
            pass

        return info

    except FileNotFoundError:
        # nvidia-smi not installed
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None



class LivePlotter:
    def __init__(self, enabled: bool = True, update_every: int = 5,
                 topo_enabled: bool = False, topo_every: int = 50,
                 topo_max_points: int = 200, topo_pca_dim: int = 30,
                 suppress_window: bool = False, plot_file: str = "training_plot.png",
                 model_info: dict = None, kelp_every: int = args.kelp_every):
        self.enabled = enabled
        self._model_info = model_info or {}
        self._wass_cbar = None  # colorbar handle for Wasserstein heatmap
        self._last_predictions = []
        self.update_every = update_every
        self._global_batch = 0
        self._ema_train = None
        self.suppress_window = suppress_window
        self.plot_file = plot_file
        self._window_closed = False
        self._reopen_requested = False
        self._lock = threading.Lock()
        self._abs_diffs_history: List[List[int]] = []

        # ── Kelp forest config ──────────────────────────────────────────
        self.kelp_every = kelp_every
        self._kelp_step = 0
        self._kelp_data = None
        self._kelp_time_offset = 0.0

        # ── TDA config ──────────────────────────────────────────────────
        self.topo_enabled = topo_enabled and _HAS_RIPSER and _HAS_GUDHI and enabled
        self.topo_every = topo_every
        self.topo_max_points = topo_max_points
        self.topo_pca_dim = topo_pca_dim
        self._topo_step = 0
        self._topo_dgms = None
        self._topo_layer_name = ""

        if topo_enabled and not _HAS_GUDHI:
            console.print("[yellow][TopoPlotter] gudhi not installed. "
                          "pip install gudhi  — TDA panels disabled.[/]")

        if not enabled:
            return

        if topo_enabled and not _HAS_RIPSER:
            console.print("[yellow][TopoPlotter] ripser not installed. "
                          "pip install ripser  — TDA panels disabled.[/]")

        _suppress_c_stderr()
        try:
            if suppress_window:
                matplotlib.use("Agg")
            else:
                matplotlib.use("TkAgg")
            self.plt = plt
        finally:
            _restore_c_stderr()

        if not suppress_window:
            self.plt.ion()

        # ── Data stores (initialized BEFORE _create_figure) ─────────────
        self.train_epoch_losses: List[float] = []
        self.val_epoch_losses: List[float] = []
        self.batch_ema: List[float] = []
        self.lr_history: List[float] = []

        self.val_batch_raw: List[float] = []
        self._val_epoch_batch_buf: List[float] = []
        self._val_epoch_avg_xs: List[float] = []
        self._val_epoch_avg_ys: List[float] = []

        self._create_figure()

        if not suppress_window:
            self._window_watch_thread = threading.Thread(
                    target=self._watch_window, daemon=True
                    )
            self._window_watch_thread.start()

    def _draw_kelp_forest(self):
        """
        Render the kelp forest visualization onto self.ax_kelp.

        Fibre-bundle interpretation (per paper §2):
          - X axis  = token index i ∈ M  (base manifold — the "sea floor")
          - Y axis  = layer depth ℓ = 0 … L  (embedding → output)
          - Each kelp strand = one fibre F_i = (V_i^(0), …, V_i^(L))
          - Within each strand, there are exactly n_layers discrete segments
          - Sway amplitude  ∝ mean activation norm   (proxy for ‖Φ^(ℓ)‖, divergence)
          - Sway frequency  ∝ std of activation norms (proxy for shear anisotropy)
          - Lean / twist    ∝ cosine drift            (proxy for Procrustes deviation / connection strength)
          - Color intensity ∝ per-token norm at that layer
          - ALL values are deterministic — derived solely from hidden_states

        No randomness anywhere. Every visual element is a function of the
        network's actual activation geometry.
        """
        ax = self.ax_kelp
        if ax is None or self._kelp_data is None:
            return

        ax.clear()
        ax.set_facecolor("#020a1a")
        ax.set_title(
                f"Kelp Forest — Fibre Bundle Dynamics  (step {self._kelp_data['step']})",
                fontsize=10, fontweight="bold", color="#c0d8e8",
                )

        data = self._kelp_data
        n_layers = data["n_layers"]       # number of hidden states (embedding + L blocks)
        n_tokens = data["n_tokens"]
        layer_stats = data["layer_stats"]
        t_anim = self._kelp_time_offset

        if n_tokens == 0 or n_layers < 1:
            ax.text(0.5, 0.5, "Not enough data",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    color="#6a9ab8", transform=ax.transAxes)
            return

        # ── Normalize stats for visual mapping ──────────────────────────
        all_mean_norms = [s["mean_norm"] for s in layer_stats]
        max_mean_norm = max(all_mean_norms) if max(all_mean_norms) > 0 else 1.0

        all_std_norms = [s["std_norm"] for s in layer_stats]
        max_std_norm = max(all_std_norms) if max(all_std_norms) > 0 else 1.0

        all_drifts = [s["cosine_drift"] for s in layer_stats]
        max_drift = max(abs(d) for d in all_drifts) if any(d != 0 for d in all_drifts) else 1.0

        # Global token norm for color normalization
        all_token_norms = []
        for s in layer_stats:
            all_token_norms.extend(s["token_norms"].tolist())
        global_max_tnorm = max(all_token_norms) if all_token_norms and max(all_token_norms) > 0 else 1.0

        # ── Layout parameters ───────────────────────────────────────────
        x_margin = 0.08
        y_floor = 0.05
        y_top = 0.92
        total_height = y_top - y_floor

        # ── Draw layer zone backgrounds ─────────────────────────────────
        # Each layer gets a band whose brightness is derived from that
        # layer's mean norm — NOT arbitrary colors.
        for li in range(n_layers):
            frac_lo = li / n_layers
            frac_hi = (li + 1) / n_layers
            y_lo = y_floor + frac_lo * total_height
            y_hi = y_floor + frac_hi * total_height

            # Brightness from mean norm of this layer (deterministic)
            norm_frac = layer_stats[li]["mean_norm"] / max_mean_norm
            # Map to a subtle blue-dark range
            r = 0.02 + 0.04 * norm_frac
            g = 0.06 + 0.06 * norm_frac
            b = 0.10 + 0.10 * norm_frac
            zone_alpha = 0.3 + 0.2 * norm_frac
            ax.axhspan(y_lo, y_hi, facecolor=(r, g, b), alpha=zone_alpha, zorder=0)

        # ── Draw sea floor (deterministic sine waves) ───────────────────
        floor_xs = np.linspace(0, 1, 200)
        floor_ys = y_floor + 0.008 * np.sin(floor_xs * 15 + t_anim * 0.3) + \
                0.004 * np.sin(floor_xs * 37 + t_anim * 0.7)
        ax.fill_between(floor_xs, 0, floor_ys, color="#1a0a30", alpha=0.8)
        ax.plot(floor_xs, floor_ys, color="#4a2a6a", linewidth=1.0, alpha=0.6)

        # ── Draw light rays (positions derived from layer stats) ────────
        # Use the per-layer mean norms to position rays deterministically
        from matplotlib.patches import Polygon as MplPolygon
        n_rays = min(4, n_layers)
        for ray_i in range(n_rays):
            # Position derived from the layer's mean norm
            stat_idx = int(ray_i * (n_layers - 1) / max(n_rays - 1, 1))
            norm_val = layer_stats[stat_idx]["mean_norm"] / max_mean_norm
            ray_x = 0.1 + ray_i * (0.8 / max(n_rays - 1, 1)) + 0.03 * np.sin(t_anim * 0.2 + norm_val * 6.28)
            ray_width = 0.02 + 0.015 * norm_val
            sway = 0.02 * np.sin(t_anim * 0.15 + ray_i * 1.618)
            ray_verts = [
                    (ray_x - ray_width, 1.0),
                    (ray_x + ray_width, 1.0),
                    (ray_x + ray_width * 2 + sway, 0.0),
                    (ray_x - ray_width * 2 + sway, 0.0),
                    ]
            ray_patch = MplPolygon(ray_verts, closed=True,
                                   facecolor="#4080b0", alpha=0.03)
            ax.add_patch(ray_patch)

        # ── Draw kelp strands (fibres) ──────────────────────────────────
        # Each strand = one fibre F_i, with EXACTLY n_layers discrete
        # segments. Each segment's geometry comes from that layer's stats.

        max_display_tokens = min(n_tokens, 32)
        if n_tokens > max_display_tokens:
            token_indices = np.linspace(0, n_tokens - 1, max_display_tokens, dtype=int)
        else:
            token_indices = np.arange(n_tokens)

        actual_spacing = (1.0 - 2 * x_margin) / max(len(token_indices) - 1, 1)

        # Sub-segments per layer for smooth curves within each layer zone
        sub_segments_per_layer = 8

        from matplotlib.collections import LineCollection

        for ti_display, ti in enumerate(token_indices):
            base_x = x_margin + ti_display * actual_spacing

            # Deterministic phase unique to this token, derived from its
            # embedding-layer norm (the actual network value, not random)
            tn_embed = layer_stats[0]["token_norms"]
            tok_embed_norm = tn_embed[min(ti, len(tn_embed) - 1)] if len(tn_embed) > 0 else 0
            phase = (tok_embed_norm / global_max_tnorm) * 6.2832  # map to [0, 2π]

            # Build path: for each layer, create sub_segments_per_layer points
            all_xs = []
            all_ys = []
            all_colors = []
            all_layer_ids = []

            for li in range(n_layers):
                mean_n = layer_stats[li]["mean_norm"]
                std_n = layer_stats[li]["std_norm"]
                drift = layer_stats[li]["cosine_drift"]

                # Per-token norm at this layer
                tn = layer_stats[li]["token_norms"]
                tok_norm = tn[min(ti, len(tn) - 1)] if len(tn) > 0 else 0

                # Sway parameters — ALL from real network values
                amplitude = (mean_n / max_mean_norm) * 0.04
                freq = 1.0 + 2.0 * (std_n / max_std_norm)
                lean = (drift / max_drift) * 0.03 if max_drift > 0 else 0

                for si in range(sub_segments_per_layer):
                    sub_frac = si / sub_segments_per_layer
                    # Global vertical fraction
                    global_frac = (li + sub_frac) / n_layers
                    y_val = y_floor + global_frac * total_height

                    # Height-dependent amplitude scaling (taller = more sway)
                    height_scale = 0.3 + 0.7 * global_frac

                    sway = (amplitude * height_scale *
                            np.sin(t_anim * freq * 0.8 + phase + global_frac * 3.0)
                            + amplitude * 0.4 * height_scale *
                            np.sin(t_anim * freq * 1.3 + phase * 2 + global_frac * 5.0)
                            + lean * global_frac *
                            np.sin(t_anim * 0.5 + phase))

                    all_xs.append(base_x + sway)
                    all_ys.append(y_val)
                    all_colors.append(tok_norm / global_max_tnorm)
                    all_layer_ids.append(li)

            # Add the final point at the top
            li_last = n_layers - 1
            tn_last = layer_stats[li_last]["token_norms"]
            tok_norm_last = tn_last[min(ti, len(tn_last) - 1)] if len(tn_last) > 0 else 0
            mean_n_last = layer_stats[li_last]["mean_norm"]
            std_n_last = layer_stats[li_last]["std_norm"]
            drift_last = layer_stats[li_last]["cosine_drift"]
            amplitude_last = (mean_n_last / max_mean_norm) * 0.04
            freq_last = 1.0 + 2.0 * (std_n_last / max_std_norm)
            lean_last = (drift_last / max_drift) * 0.03 if max_drift > 0 else 0

            sway_top = (amplitude_last * np.sin(t_anim * freq_last * 0.8 + phase + 3.0)
                        + amplitude_last * 0.4 * np.sin(t_anim * freq_last * 1.3 + phase * 2 + 5.0)
                        + lean_last * np.sin(t_anim * 0.5 + phase))
            all_xs.append(base_x + sway_top)
            all_ys.append(y_top)
            all_colors.append(tok_norm_last / global_max_tnorm)
            all_layer_ids.append(li_last)

            all_xs = np.array(all_xs)
            all_ys = np.array(all_ys)
            all_colors = np.array(all_colors)
            all_layer_ids = np.array(all_layer_ids)

            # Build line segments
            points = np.array([all_xs, all_ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color: green hue, brightness from token norm at that layer
            # Per-layer hue shift derived from that layer's cosine drift
            colors_rgba = []
            for ci in range(len(segments)):
                intensity = 0.25 + 0.75 * all_colors[ci]
                li_ci = all_layer_ids[ci]
                # Hue shift from cosine drift (deterministic, from network)
                drift_ci = layer_stats[li_ci]["cosine_drift"]
                drift_norm = abs(drift_ci) / max_drift if max_drift > 0 else 0
                r = 0.05 + 0.12 * drift_norm
                g = 0.15 + 0.35 * intensity
                b = 0.10 + 0.15 * drift_norm
                a = 0.4 + 0.5 * intensity
                colors_rgba.append((r, g, b, a))

            # Strand thickness: thicker at base, thinner at top
            fracs = np.linspace(0, 1, len(segments))
            linewidths = 2.5 - 1.5 * fracs

            lc = LineCollection(segments, colors=colors_rgba, linewidths=linewidths,
                                capstyle="round", joinstyle="round")
            ax.add_collection(lc)

            # ── Draw nodes at EACH layer boundary ───────────────────────
            # These are the discrete fibre morphism points Φ^(ℓ)
            for li in range(n_layers):
                # Find the exact position of this layer boundary on the strand
                seg_idx = li * sub_segments_per_layer
                seg_idx = min(seg_idx, len(all_xs) - 1)
                node_x = all_xs[seg_idx]
                node_y = all_ys[seg_idx]

                # Node size from layer's mean norm
                node_size = 2.0 + 3.0 * (layer_stats[li]["mean_norm"] / max_mean_norm)

                # Color from per-token norm at this layer
                tn_li = layer_stats[li]["token_norms"]
                tok_n = tn_li[min(ti, len(tn_li) - 1)] if len(tn_li) > 0 else 0
                node_intensity = tok_n / global_max_tnorm

                node_color = (
                        0.2 + 0.3 * node_intensity,
                        0.5 + 0.4 * node_intensity,
                        0.3 + 0.2 * node_intensity,
                        0.6 + 0.3 * node_intensity,
                        )
                ax.plot(node_x, node_y, 'o',
                        color=node_color, markersize=node_size,
                        markeredgecolor=(0.4, 0.8, 0.5, 0.3),
                        markeredgewidth=0.5, zorder=4)

            # ── Draw fronds at layer midpoints (deterministic) ──────────
            # One frond per layer, positioned at the midpoint of each
            # layer zone. Direction derived from cosine drift sign.
            for li in range(n_layers):
                mid_seg = li * sub_segments_per_layer + sub_segments_per_layer // 2
                mid_seg = min(mid_seg, len(all_xs) - 1)
                fx = all_xs[mid_seg]
                fy = all_ys[mid_seg]

                drift_li = layer_stats[li]["cosine_drift"]
                # Frond direction: sign of drift determines left/right
                frond_sign = 1.0 if drift_li >= 0 else -1.0
                # Alternate sides for even/odd tokens
                if ti_display % 2 == 1:
                    frond_sign *= -1.0

                frond_angle = frond_sign * (0.4 + 0.3 * abs(drift_li) / max_drift) if max_drift > 0 else frond_sign * 0.4
                frond_angle += 0.1 * np.sin(t_anim * 0.7 + phase + li)

                # Frond length from std norm (more variation = longer frond)
                frond_len = 0.012 + 0.01 * (layer_stats[li]["std_norm"] / max_std_norm)
                frond_dx = np.cos(frond_angle) * frond_len
                frond_dy = np.sin(frond_angle) * frond_len

                frond_intensity = layer_stats[li]["mean_norm"] / max_mean_norm
                ax.plot([fx, fx + frond_dx], [fy, fy + frond_dy],
                        color=(0.10 + 0.08 * frond_intensity,
                               0.35 + 0.15 * frond_intensity,
                               0.20 + 0.08 * frond_intensity,
                               0.35 + 0.2 * frond_intensity),
                        linewidth=1.0)

            # ── Root anchor ─────────────────────────────────────────────
            from matplotlib.patches import Ellipse
            anchor = Ellipse((base_x, y_floor), width=0.012, height=0.008,
                             facecolor="#2a1540", edgecolor="#4a2a6a",
                             linewidth=0.5, alpha=0.7)
            ax.add_patch(anchor)

        # ── Layer boundary lines ────────────────────────────────────────
        for li in range(n_layers + 1):
            frac = li / n_layers
            y_line = y_floor + frac * total_height

            if li == 0 or li == n_layers:
                ax.axhline(y=y_line, color="#5a8aaa", linewidth=1.0,
                           linestyle="-", alpha=0.4, zorder=2)
            else:
                ax.axhline(y=y_line, color="#4a7a9a", linewidth=0.8,
                           linestyle="--", alpha=0.35, zorder=2)

            # Layer labels
            if li < n_layers:
                label_y = y_floor + (li + 0.5) / n_layers * total_height
                if li == 0:
                    label = "Embed"
                elif li == n_layers - 1:
                    label = "Output"
                else:
                    label = f"L{li}"

                ax.text(0.995, label_y, label,
                        fontsize=7, color="#6a9aba", alpha=0.6,
                        ha="right", va="center", transform=ax.get_yaxis_transform(),
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a1a2a",
                                  edgecolor="none", alpha=0.5))

        # ── Water caustics (deterministic, from layer stats) ────────────
        # One caustic per layer, position and size from that layer's stats
        from matplotlib.patches import Circle
        for li in range(min(n_layers, 8)):
            norm_frac = layer_stats[li]["mean_norm"] / max_mean_norm
            std_frac = layer_stats[li]["std_norm"] / max_std_norm
            cx = 0.1 + li * (0.8 / max(n_layers - 1, 1)) + 0.03 * np.sin(t_anim * 0.4 + norm_frac * 6.28)
            cy = y_floor + (li + 0.5) / n_layers * total_height
            caustic_alpha = 0.01 + 0.015 * std_frac
            caustic_radius = 0.04 + 0.03 * norm_frac
            caustic = Circle((cx, cy), radius=caustic_radius,
                             facecolor="#60b0d0",
                             alpha=max(0, min(caustic_alpha, 0.03)),
                             edgecolor="none")
            ax.add_patch(caustic)

        # ── Floating particles (deterministic, from per-token norms) ────
        # Instead of random particles, place one "spore" per token per
        # layer, at a position derived from that token's norm at that layer.
        # This makes every particle a real data point.
        n_display = len(token_indices)
        for li in range(n_layers):
            tn_li = layer_stats[li]["token_norms"]
            for ti_d, ti in enumerate(token_indices):
                tok_n = tn_li[min(ti, len(tn_li) - 1)] if len(tn_li) > 0 else 0
                norm_frac = tok_n / global_max_tnorm

                # Position: offset from the strand, derived from norm
                # Small deterministic displacement so particles don't overlap strands
                px = x_margin + ti_d * actual_spacing + 0.015 * np.sin(norm_frac * 6.28 + li * 1.618)
                py = y_floor + (li + 0.5) / n_layers * total_height + 0.005 * np.cos(norm_frac * 6.28 + ti * 1.618)
                py = np.clip(py, y_floor, y_top)

                p_alpha = 0.08 + 0.15 * norm_frac
                p_size = 0.3 + 1.2 * norm_frac

                ax.plot(px, py, '.', color="#80c0e0",
                        alpha=min(p_alpha, 0.25), markersize=p_size, zorder=5)

        # ── Stats annotation ────────────────────────────────────────────
        mean_norms_str = ", ".join(f"{s['mean_norm']:.1f}" for s in layer_stats[:6])
        if n_layers > 6:
            mean_norms_str += " …"
        drift_str = ", ".join(f"{s['cosine_drift']:.3f}" for s in layer_stats[:6])
        if n_layers > 6:
            drift_str += " …"

        ax.text(0.01, 0.98,
                f"Layers: {n_layers}  Tokens: {n_tokens}  Fibres: {len(token_indices)}\n"
                f"Mean‖h‖: [{mean_norms_str}]\n"
                f"Drift:   [{drift_str}]",
                fontsize=5.5, color="#5a8aaa", alpha=0.5,
                ha="left", va="top", transform=ax.transAxes,
                fontfamily="monospace")

        # ── Axis limits and cleanup ─────────────────────────────────────
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([])
        ax.set_yticks([])


    @torch.no_grad()
    def update_kelp_forest(self, model: nn.Module, input_ids: torch.Tensor):
        """
        Extract hidden-state statistics from the model and render the kelp forest.

        Each kelp strand = one layer's embedding space.
        The strand is rooted at the "index manifold" (bottom axis = token position).
        Sway is driven by per-layer statistics:
          - mean activation magnitude  → lateral sway amplitude
          - std of activations         → sway frequency multiplier
          - cross-layer cosine drift   → additional twist
        """
        if not self.enabled:
            return

        self._kelp_step += 1
        if self._kelp_step != 1 and self._kelp_step % self.kelp_every != 0:
            return

        was_training = model.training
        model.eval()

        try:
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) < 2:
                return

            # Gather per-layer statistics
            layer_stats = []
            prev_mean_vec = None
            for li, hs in enumerate(hidden_states):
                # hs shape: (B, T, D)
                flat = hs.float().reshape(-1, hs.shape[-1])  # (B*T, D)
                norms = flat.norm(dim=-1)  # (B*T,)
                mean_norm = norms.mean().item()
                std_norm = norms.std().item()
                mean_vec = flat.mean(dim=0)  # (D,)

                # Cosine similarity to previous layer's mean vector
                cosine_drift = 0.0
                if prev_mean_vec is not None:
                    cos = F.cosine_similarity(
                            mean_vec.unsqueeze(0), prev_mean_vec.unsqueeze(0)
                            ).item()
                    cosine_drift = 1.0 - cos  # 0 = identical, ~2 = opposite

                # Per-token norms for this layer (take first batch element)
                token_norms = hs[0].float().norm(dim=-1).cpu().numpy()  # (T,)

                layer_stats.append({
                    "mean_norm": mean_norm,
                    "std_norm": std_norm,
                    "cosine_drift": cosine_drift,
                    "token_norms": token_norms,
                    })
                prev_mean_vec = mean_vec

            self._kelp_data = {
                    "layer_stats": layer_stats,
                    "n_layers": len(hidden_states),
                    "n_tokens": hidden_states[0].shape[1],
                    "step": self._kelp_step,
                    }

            # Advance the sway "time" so animation progresses between updates
            self._kelp_time_offset += 1.0

            self._draw_kelp_forest()
            self._refresh()

        except Exception:
            pass
        finally:
            if was_training:
                model.train()


    def accumulate_predictions(self, predictions: list):
        """Accumulate predictions across batches (call clear_predictions() at epoch start)."""
        if not self.enabled:
            return
        self._last_predictions.extend(predictions)
        # Keep only the most recent 20 for display
        if len(self._last_predictions) > 20:
            self._last_predictions = self._last_predictions[-20:]
        self._draw_predictions()
        self._refresh()

    def clear_predictions(self):
        """Clear accumulated predictions (call at start of each epoch)."""
        self._last_predictions = []

    def _create_figure(self):
        """Create (or recreate) the matplotlib figure and axes."""

        if self.topo_enabled:
            self.fig = self.plt.figure(figsize=(26, 22))
            gs = self.fig.add_gridspec(4, 3, hspace=0.50, wspace=0.35)

            ax_epoch   = self.fig.add_subplot(gs[0, 0])
            ax_batch   = self.fig.add_subplot(gs[0, 1])
            ax_lr      = self.fig.add_subplot(gs[0, 2])
            ax_val     = self.fig.add_subplot(gs[1, 0])
            ax_barcode = self.fig.add_subplot(gs[1, 1])
            ax_bd      = self.fig.add_subplot(gs[1, 2])
            ax_kelp    = self.fig.add_subplot(gs[2, 0:2])
            ax_diffs   = self.fig.add_subplot(gs[2, 2])
            ax_preds   = self.fig.add_subplot(gs[3, 0:2])
            ax_info    = self.fig.add_subplot(gs[3, 2])

            self.ax_barcode = ax_barcode
            self.ax_bd = ax_bd

            ax_diffs.set_title("Prediction Error Score",
                               fontsize=10, fontweight="bold")
            ax_diffs.set_xlabel("Prediction Update #")
            ax_diffs.set_ylabel("Error Score (0 = perfect, 1 = garbage)")
            ax_diffs.axhline(y=0, color="green", linewidth=0.8, linestyle="-", alpha=0.4, label="_nolegend_")
            ax_diffs.axhline(y=1, color="red", linewidth=0.8, linestyle="-", alpha=0.4, label="_nolegend_")
            ax_diffs.set_ylim(-0.05, 1.05)
            ax_diffs.grid(True, alpha=0.3)

            ax_diffs.legend(loc="lower left", fontsize=7, framealpha=0.7)

            self.ax_diffs = ax_diffs
            self._scatter_diffs = None

        else:
            self.fig = self.plt.figure(figsize=(24, 18))
            gs = self.fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

            ax_epoch = self.fig.add_subplot(gs[0, 0])
            ax_batch = self.fig.add_subplot(gs[0, 1])
            ax_lr    = self.fig.add_subplot(gs[0, 2])
            ax_val   = self.fig.add_subplot(gs[1, 0])
            ax_diffs = self.fig.add_subplot(gs[1, 1])
            ax_kelp  = self.fig.add_subplot(gs[1, 2])
            ax_preds = self.fig.add_subplot(gs[2, 0:2])
            ax_info  = self.fig.add_subplot(gs[2, 2])

            ax_diffs.set_title("Prediction Error Score",
                               fontsize=10, fontweight="bold")
            ax_diffs.set_xlabel("Prediction Update #")
            ax_diffs.set_ylabel("Error Score (0 = perfect, 1 = garbage)")
            ax_diffs.axhline(y=0, color="green", linewidth=0.8, linestyle="-", alpha=0.4, label="_nolegend_")
            ax_diffs.axhline(y=1, color="red", linewidth=0.8, linestyle="-", alpha=0.4, label="_nolegend_")
            ax_diffs.set_ylim(-0.05, 1.05)
            ax_diffs.grid(True, alpha=0.3)

            ax_diffs.legend(loc="lower left", fontsize=7, framealpha=0.7)

            self.ax_diffs = ax_diffs
            self._scatter_diffs = None

            self.ax_barcode = None
            self.ax_bd = None

        # ── Kelp forest axes ────────────────────────────────────────────────
        self.ax_kelp = ax_kelp
        ax_kelp.set_title("Kelp Forest — Embedding Space Dynamics",
                          fontsize=10, fontweight="bold")
        ax_kelp.set_facecolor("#020a1a")
        ax_kelp.set_xlim(0, 1)
        ax_kelp.set_ylim(0, 1)
        ax_kelp.set_xticks([])
        ax_kelp.set_yticks([])
        ax_kelp.text(0.5, 0.5, "Waiting for hidden states...",
                     ha="center", va="center", fontsize=11, alpha=0.4,
                     color="#6a9ab8", transform=ax_kelp.transAxes)

        # Store references to the text-only axes
        self.ax_preds = ax_preds
        self.ax_info = ax_info

        # _plot_axes: only axes that have actual data lines
        self._plot_axes = [ax_epoch, ax_batch, ax_lr, ax_val]
        if hasattr(self, 'ax_diffs') and self.ax_diffs is not None:
            self._plot_axes.append(self.ax_diffs)
        if self.ax_barcode is not None:
            self._plot_axes.append(self.ax_barcode)
        if self.ax_bd is not None:
            self._plot_axes.append(self.ax_bd)

        self.axes = np.array(self._plot_axes)

        # ── Window title ────────────────────────────────────────────────────
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("LLVM IR GPT — Live Training")
        except Exception:
            pass

        # ── Top-left: Epoch losses ──────────────────────────────────────────
        ax_epoch.set_title("Epoch Loss", fontsize=10, fontweight="bold")
        ax_epoch.set_xlabel("Epoch")
        ax_epoch.set_ylabel("Loss")
        ax_epoch.grid(True, alpha=0.3)
        self.line_train_epoch, = ax_epoch.plot(
                [], [], label="Train", color="steelblue", linewidth=2,
                )
        self.line_val_epoch, = ax_epoch.plot(
                [], [], label="Val", color="tomato", linewidth=2,
                )
        ax_epoch.legend(loc="upper right", fontsize=8)

        # ── Top-center: Batch loss EMA ──────────────────────────────────────
        ax_batch.set_title("Batch Loss (Train EMA)", fontsize=10, fontweight="bold")
        ax_batch.set_xlabel("Batch")
        ax_batch.set_ylabel("Loss")
        ax_batch.grid(True, alpha=0.3)
        self.line_batch_ema, = ax_batch.plot(
                [], [], label="Train EMA", color="steelblue", linewidth=2,
                )
        ax_batch.legend(loc="upper right", fontsize=8)

        # ── Top-right: Learning Rate ────────────────────────────────────────
        ax_lr.set_title("Learning Rate", fontsize=10, fontweight="bold")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
        self.line_lr, = ax_lr.plot(
                [], [], label="LR", color="seagreen", linewidth=2,
                )
        ax_lr.legend(loc="upper right", fontsize=8)

        # ── Mid-left: Val batch loss ────────────────────────────────────────
        ax_val.set_title("Batch Loss (Val)", fontsize=10, fontweight="bold")
        ax_val.set_xlabel("Global Val Batch")
        ax_val.set_ylabel("Loss")
        ax_val.grid(True, alpha=0.3)
        self.line_val_raw, = ax_val.plot(
                [], [], label="Val Batch", color="tomato", linewidth=1.0, alpha=0.5,
                )
        self.line_val_epoch_avg, = ax_val.plot(
                [], [], label="Val Epoch Avg", color="darkred", linewidth=2.0,
                marker="o", markersize=4,
                )
        ax_val.legend(loc="upper right", fontsize=8)

        # ── TDA panels (only when topo_enabled) ─────────────────────────────
        if self.ax_barcode is not None:
            self.ax_barcode.set_title("Persistence Landscapes (H₁, all layers)",
                                      fontsize=10, fontweight="bold")
            self.ax_barcode.text(
                    0.5, 0.5, "Waiting for data...",
                    ha="center", va="center",
                    transform=self.ax_barcode.transAxes,
                    fontsize=11, alpha=0.4,
                    )
            self.ax_barcode.grid(True, alpha=0.2)

        if self.ax_bd is not None:
            self.ax_bd.set_title("Wasserstein-1 Distance Heatmap (layers × layers)",
                                 fontsize=10, fontweight="bold")
            # Create a dummy 1x1 image and a PERMANENT colorbar
            self._wass_im = self.ax_bd.imshow(
                    np.zeros((1, 1)),
                    cmap="inferno",
                    interpolation="nearest",
                    origin="lower",
                    aspect="equal",
                    vmin=0.0,
                    vmax=1.0,
                    )
            self._wass_cbar = self.fig.colorbar(
                    self._wass_im, ax=self.ax_bd, fraction=0.046, pad=0.04,
                    )
            self._wass_cbar.ax.tick_params(labelsize=6)
            self._wass_cbar.set_label("Relative W₁", fontsize=7)
            # Store the axes position so we can restore it if needed
            self._wass_ax_pos = self.ax_bd.get_position()


        # ── Predictions panel (text only) ───────────────────────────────────
        ax_preds.set_xlim(0, 1)
        ax_preds.set_ylim(0, 1)
        ax_preds.axis("off")
        ax_preds.set_title(
                "Last Batch Predictions (Expected → Got)",
                fontsize=10, fontweight="bold",
                )
        self._draw_predictions()

        # ── Model Info panel (text only) ────────────────────────────────────
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        ax_info.set_title("Model / System Info", fontsize=10, fontweight="bold")
        self._draw_model_info()

        # ── Layout ──────────────────────────────────────────────────────────
        self.fig.tight_layout()

        # ── Restore existing data onto the new figure ───────────────────────
        self._restore_data()

        # ── Register close event ────────────────────────────────────────────
        if not self.suppress_window:
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        # ── Force window to appear ──────────────────────────────────────────
        if not self.suppress_window:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def _restore_data(self):
        """Re-plot all accumulated data onto current line objects."""
        # ... existing restore code ...

        self._draw_predictions()
        self._draw_model_info()

        # Restore diff plot (scatter only, last 1000)
        if hasattr(self, 'ax_diffs') and self.ax_diffs is not None and self._abs_diffs_history:
            n_updates = len(self._abs_diffs_history)
            max_scatter_window = 1000

            scatter_start = max(0, n_updates - max_scatter_window)
            all_scatter_x = []
            all_scatter_y = []
            for i in range(scatter_start, n_updates):
                for s in self._abs_diffs_history[i]:
                    all_scatter_x.append(i)
                    all_scatter_y.append(s)

            if all_scatter_x:
                n_scatter = len(all_scatter_x)
                scatter_alpha = 0.30 if n_scatter < 500 else (0.15 if n_scatter < 2000 else 0.08)
                scatter_size = 14 if n_scatter < 500 else (10 if n_scatter < 2000 else 6)
                self._scatter_diffs = self.ax_diffs.scatter(
                        all_scatter_x, all_scatter_y,
                        s=scatter_size, alpha=scatter_alpha, color="steelblue", zorder=1,
                        edgecolors="none",
                        )

            x_lo = max(scatter_start - 0.5, -0.5)
            self.ax_diffs.set_xlim(x_lo, max(n_updates - 0.5, 0.5))
            self.ax_diffs.set_ylim(-0.05, 1.05)

            from matplotlib.lines import Line2D
            window_label = f"Individual (last {min(max_scatter_window, n_updates)})"
            legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                           markersize=6, alpha=0.5, linestyle='None', label=window_label),
                    ]
            self.ax_diffs.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.7)

    def _on_close(self, event):
        """Called when the user closes the matplotlib window."""
        with self._lock:
            self._window_closed = True
        console.print(
                "\n[bold yellow]📊 Plot window closed. "
                "Press [bold white]r[/bold white] to reopen it.[/]\n"
                )

    def _watch_window(self):
        """Background thread: periodically pumps the matplotlib event loop
        so that close events are detected even between batch updates."""
        while True:
            time.sleep(0.5)
            # We do NOT touch the flags here.
            # The main thread's _check_reopen() is the sole consumer.

    def request_reopen(self):
        """Call this (e.g. from EpochController) to request window reopen."""
        with self._lock:
            if self._window_closed:
                self._reopen_requested = True

    def _check_reopen(self):
        """Check if we need to reopen the window. Must be called from main thread."""
        with self._lock:
            needs_reopen = self._reopen_requested and self._window_closed
            if needs_reopen:
                self._reopen_requested = False
                self._window_closed = False

        if needs_reopen:
            console.print("[bold green]📊 Reopening plot window...[/]")
            self.plt.ion()
            self._create_figure()

    def _is_window_alive(self) -> bool:
        """Check if the figure window is still open."""
        with self._lock:
            return not self._window_closed

    # ── Save figure to file ─────────────────────────────────────────────
    def _save_to_file(self):
        """Save the current figure to disk."""
        if not self.enabled:
            return
        try:
            save_path = self.plot_file
            if run_dir is not None:
                save_path = os.path.join(run_dir, self.plot_file)
            self.fig.savefig(save_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            pass  # Don't crash training for a file write error

    # ── Refresh helper ──────────────────────────────────────────────────
    def _refresh(self):
        if not self.enabled:
            return

        if not self.suppress_window:
            self._check_reopen()

        _suppress_c_stderr()
        try:
            for ax in self._plot_axes:
                if ax is self.ax_diffs:
                    # Skip relim/autoscale entirely for ax_diffs —
                    # we manage its limits manually in update_prediction_diffs
                    continue
                ax.relim()
                ax.autoscale_view()

            if not self.suppress_window and self._is_window_alive():
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            else:
                self.fig.canvas.draw()
        finally:
            _restore_c_stderr()


    # ── Batch update (train) ────────────────────────────────────────────
    def update_batch(self, batch_loss: float):
        if not self.enabled:
            return
        self._global_batch += 1

        alpha = 0.05
        if self._ema_train is None:
            self._ema_train = batch_loss
        else:
            self._ema_train = alpha * batch_loss + (1 - alpha) * self._ema_train
        self.batch_ema.append(self._ema_train)

        if self._global_batch % self.update_every == 0:
            xs = list(range(len(self.batch_ema)))
            self.line_batch_ema.set_data(xs, self.batch_ema)
            self._refresh()

    # ── Val batch ───────────────────────────────────────────────────────
    def update_val_batch(self, val_batch_loss: float):
        if not self.enabled:
            return
        self.val_batch_raw.append(val_batch_loss)
        self._val_epoch_batch_buf.append(val_batch_loss)

    def finish_val_epoch(self):
        if not self.enabled:
            return
        if self._val_epoch_batch_buf:
            avg = sum(self._val_epoch_batch_buf) / len(self._val_epoch_batch_buf)
            end_x = len(self.val_batch_raw) - 1
            start_x = end_x - len(self._val_epoch_batch_buf) + 1
            mid_x = (start_x + end_x) / 2.0
            self._val_epoch_avg_xs.append(mid_x)
            self._val_epoch_avg_ys.append(avg)
            self._val_epoch_batch_buf = []

        vxs = list(range(len(self.val_batch_raw)))
        self.line_val_raw.set_data(vxs, self.val_batch_raw)
        self.line_val_epoch_avg.set_data(self._val_epoch_avg_xs,
                                         self._val_epoch_avg_ys)

    # ── Epoch update ────────────────────────────────────────────────────
    def update_epoch(self, train_loss: float, val_loss: float, lr: float):
        if not self.enabled:
            return

        self.train_epoch_losses.append(train_loss)
        self.val_epoch_losses.append(val_loss)
        self.lr_history.append(lr)

        epochs = list(range(1, len(self.train_epoch_losses) + 1))
        self.line_train_epoch.set_data(epochs, self.train_epoch_losses)
        self.line_val_epoch.set_data(epochs, self.val_epoch_losses)
        self.line_lr.set_data(epochs, self.lr_history)

        self._refresh()

        # Always save to file after each epoch
        self._save_to_file()

    # ════════════════════════════════════════════════════════════════════
    # TDA methods (unchanged from original)
    # ════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def update_topo(self, model: nn.Module, input_ids: torch.Tensor):
        """
        Compute per-layer persistence landscapes and cross-layer Wasserstein
        heatmap, then render them onto the two TDA panels.
        """
        if not self.topo_enabled:
            return

        self._topo_step += 1
        if self._topo_step != 1 and self._topo_step % self.topo_every != 0:
            return

        was_training = model.training
        model.eval()

        try:
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states
            if hidden_states is None or len(hidden_states) < 2:
                return

            # ── Persistence Landscapes (replaces barcode) ───────────────
            n_landscapes = 5
            resolution = 100
            landscapes, sample_range = compute_persistence_landscapes(
                    hidden_states,
                    n_landscapes=n_landscapes,
                    resolution=resolution,
                    max_points=self.topo_max_points,
                    pca_dim=self.topo_pca_dim,
                    )
            self._draw_persistence_landscapes(
                    landscapes, sample_range, n_landscapes, resolution,
                    )

            # ── Wasserstein Heatmap (replaces birth/death) ──────────────
            W = compute_cross_layer_wasserstein(
                    hidden_states,
                    max_points=self.topo_max_points,
                    pca_dim=self.topo_pca_dim,
                    )
            self._draw_wasserstein_heatmap(W)

            self._refresh()

        except Exception as e:
            # Silently skip — don't crash training for a viz failure
            import traceback
            traceback.print_exc()
        finally:
            if was_training:
                model.train()


    def update_prediction_diffs(self, predictions: list):
        """
        Compute a per-sample error score in [0, 1] and update the diff plot.
        Only scatter points are shown (last ~1000 updates).
        """
        if not self.enabled:
            return

        scores = []
        scale = 0.1

        for expected, predicted, _ in predictions:
            try:
                exp_val = int(expected.strip())
            except (ValueError, TypeError, AttributeError):
                continue

            pred_cleaned = ''.join(
                    c for c in predicted if c.isascii() and c.isprintable()
                    ).strip()

            try:
                pred_val = int(pred_cleaned)
            except (ValueError, TypeError):
                scores.append(1.0)
                continue

            diff = abs(exp_val - pred_val)
            if diff == 0:
                scores.append(0.0)
            else:
                scores.append(0.9 * math.tanh(scale * diff))

        if not scores:
            return

        self._abs_diffs_history.append(scores)

        if not hasattr(self, 'ax_diffs') or self.ax_diffs is None:
            return

        ax = self.ax_diffs
        n_updates = len(self._abs_diffs_history)

        # ── Only show the last ~1000 updates as scatter ─────────────────
        max_scatter_window = 1000

        # ── Remove old scatter ──────────────────────────────────────────
        if self._scatter_diffs is not None:
            try:
                self._scatter_diffs.remove()
            except (ValueError, AttributeError):
                pass
            self._scatter_diffs = None

        # ── Build new scatter data (WINDOWED to last 1000) ──────────────
        scatter_start = max(0, n_updates - max_scatter_window)
        all_scatter_x = []
        all_scatter_y = []
        for i in range(scatter_start, n_updates):
            for s in self._abs_diffs_history[i]:
                all_scatter_x.append(i)
                all_scatter_y.append(s)

        # Adaptive alpha
        n_scatter = len(all_scatter_x)
        if n_scatter > 5000:
            scatter_alpha = 0.08
            scatter_size = 6
        elif n_scatter > 2000:
            scatter_alpha = 0.15
            scatter_size = 10
        elif n_scatter > 500:
            scatter_alpha = 0.22
            scatter_size = 12
        else:
            scatter_alpha = 0.30
            scatter_size = 14

        if all_scatter_x:
            self._scatter_diffs = ax.scatter(
                    all_scatter_x, all_scatter_y,
                    s=scatter_size, alpha=scatter_alpha, color="steelblue", zorder=1,
                    edgecolors="none",
                    )

        # ── Set axis limits ─────────────────────────────────────────────
        x_lo = max(scatter_start - 0.5, -0.5)
        x_hi = max(n_updates - 0.5, 0.5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-0.05, 1.05)

        # ── Update legend (scatter only) ────────────────────────────────
        from matplotlib.lines import Line2D
        window_label = f"Individual (last {min(max_scatter_window, n_updates)})"
        legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                       markersize=6, alpha=0.5, linestyle='None', label=window_label),
                ]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=7, framealpha=0.7)

        self._refresh()

    @staticmethod
    def _downsample_line_indices(
            n_total: int,
            max_points: int = 1500,
            recent_full_res: int = 300,
            ) -> List[int]:
        """
        Two-tier downsampling strategy for line data:
          - The most recent `recent_full_res` points are kept at full resolution.
          - Older points are uniformly downsampled so the total never exceeds
            `max_points`.
          - The first and last points of the old region are always included
            to avoid visual jumps.

        This keeps recent trends sharp while preventing matplotlib from
        choking on tens of thousands of line segments.
        """
        if n_total <= max_points:
            return list(range(n_total))

        # Split into old region and recent region
        recent_start = max(0, n_total - recent_full_res)
        n_recent = n_total - recent_start
        budget_for_old = max_points - n_recent

        if recent_start == 0 or budget_for_old <= 2:
            # Everything is "recent" or no budget for old
            # Just uniformly downsample everything
            step = max(1, n_total // max_points)
            indices = list(range(0, n_total, step))
            if indices[-1] != n_total - 1:
                indices.append(n_total - 1)
            return indices

        # Old region: [0, recent_start) — downsample to budget_for_old points
        old_step = max(1, recent_start // budget_for_old)
        old_indices = list(range(0, recent_start, old_step))
        # Ensure the boundary point is included for continuity
        if old_indices[-1] != recent_start - 1:
            old_indices.append(recent_start - 1)

        # Recent region: [recent_start, n_total) — full resolution
        recent_indices = list(range(recent_start, n_total))

        return old_indices + recent_indices

    def _draw_predictions(self):
        """Draw the last batch's expected vs predicted with differences."""
        ax = self.ax_preds
        if ax is None:
            return
        ax.clear()
        ax.axis("off")
        ax.set_title(
                "Predictions (Training + Validation samples)",
                fontsize=10, fontweight="bold",
                )

        if not self._last_predictions:
            ax.text(0.5, 0.5, "Waiting for predictions...",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        n_show = min(len(self._last_predictions), 10)
        y_positions = np.linspace(0.95, 0.05, n_show)

        for i, (expected, predicted, is_correct) in enumerate(
                self._last_predictions[-n_show:]
                ):
            # Try to parse both as integers
            exp_parseable = True
            pred_parseable = True
            try:
                exp_val = int(expected.strip())
            except (ValueError, TypeError):
                exp_parseable = False

            # Clean BPE artifacts before parsing
            pred_cleaned = ''.join(
                    c for c in predicted if c.isascii() and c.isprintable()
                    ).strip()

            try:
                pred_val = int(pred_cleaned)
            except (ValueError, TypeError):
                pred_parseable = False

            # ── Determine color and marker from NUMERIC comparison ──────
            if exp_parseable and pred_parseable:
                diff = abs(exp_val - pred_val)
                if diff == 0:
                    color = "green"
                    marker = "✓"
                elif diff <= 5:
                    color = "orange"
                    marker = "≈"
                elif diff <= 50:
                    color = "darkorange"
                    marker = "✗"
                else:
                    color = "red"
                    marker = "✗"

                text = (
                        f"{marker}  expected: {exp_val:>6d}  │  "
                        f"got: {pred_val:>6d}  │  diff: {diff}"
                        )

            elif exp_parseable and not pred_parseable:
                color = "red"
                marker = "✗"
                pred_display = pred_cleaned[:20] if pred_cleaned else predicted.strip()[:20]
                if len(pred_display) > 20:
                    pred_display = pred_display[:20] + "…"
                text = (
                        f"{marker}  expected: {exp_val:>6d}  │  "
                        f"got: {pred_display:<20s}  │  ⚠ UNPARSEABLE"
                        )

            else:
                color = "red"
                marker = "✗"
                text = (
                        f"{marker}  expected: {expected[:12]:>12s}  │  "
                        f"got: {predicted[:12]:>12s}  │  ⚠ BOTH INVALID"
                        )

            ax.text(0.02, y_positions[i], text,
                    fontsize=8, fontfamily="monospace",
                    color=color, transform=ax.transAxes,
                    verticalalignment="center")

        # Summary stats — use numeric comparison
        n_correct = sum(
                1 for exp, pred, _ in self._last_predictions
                if _is_int_str(exp) and _is_int_str(
                    ''.join(c for c in pred if c.isascii() and c.isprintable()).strip()
                    )
                and int(exp.strip()) == int(
                    ''.join(c for c in pred if c.isascii() and c.isprintable()).strip()
                    )
                )
        n_total = len(self._last_predictions)
        n_garbage = sum(
                1 for _, pred, _ in self._last_predictions
                if not _is_int_str(
                    ''.join(c for c in pred if c.isascii() and c.isprintable()).strip()
                    )
                )
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0

        summary = f"Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)"
        if n_garbage > 0:
            summary += f"  │  ⚠ {n_garbage} unparseable"

        ax.text(0.98, 0.01, summary,
                ha="right", va="bottom", fontsize=9, fontweight="bold",
                transform=ax.transAxes, alpha=0.7)

    def _draw_model_info(self):
        """Draw basic model information + GPU/system stats."""
        ax = self.ax_info
        if ax is None:
            return
        ax.clear()
        ax.axis("off")
        ax.set_title("Model / System Info", fontsize=10, fontweight="bold")

        if not self._model_info:
            ax.text(0.5, 0.5, "No model info available",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        info_lines = [
                f"Parameters:    {self._model_info.get('params', '?'):,}",
                f"d_model:       {self._model_info.get('d_model', '?')}",
                f"n_heads:       {self._model_info.get('n_heads', '?')}",
                f"n_layers:      {self._model_info.get('n_layers', '?')}",
                f"max_seq_len:   {self._model_info.get('max_seq_len', '?')}",
                f"vocab_size:    {self._model_info.get('vocab_size', '?')}",
                f"device:        {self._model_info.get('device', '?')}",
                ]

        # Add GPU info if available, otherwise show CPU/system info
        gpu = get_gpu_info()
        info_lines.append(f"───────────────────────────")
        if gpu:
            if "gpu_name" in gpu:
                info_lines.append(f"GPU:           {gpu['gpu_name']}")
            info_lines.append(
                    f"GPU Mem:       {gpu.get('gpu_mem_used_mb', '?')} / "
                    f"{gpu.get('gpu_mem_total_mb', '?')} MB "
                    f"({gpu.get('gpu_mem_free_mb', '?')} MB free)"
                    )
            info_lines.append(f"GPU Util:      {gpu.get('gpu_util_pct', '?')}%")
            if "gpu_temp_c" in gpu:
                info_lines.append(f"GPU Temp:      {gpu['gpu_temp_c']}°C")
            if "gpu_power_w" in gpu:
                info_lines.append(f"GPU Power:     {gpu['gpu_power_w']} W")
        else:
            # No nvidia-smi — show CPU/system info instead
            import platform
            info_lines.append(f"GPU:           N/A (no nvidia-smi)")
            if torch.cuda.is_available():
                info_lines.append(f"CUDA:          {torch.version.cuda}")
                info_lines.append(
                        f"CUDA Device:   {torch.cuda.get_device_name(0)}"
                        )
                mem = torch.cuda.mem_get_info(0)
                free_mb = mem[0] // (1024 * 1024)
                total_mb = mem[1] // (1024 * 1024)
                used_mb = total_mb - free_mb
                info_lines.append(
                        f"CUDA Mem:      {used_mb} / {total_mb} MB "
                        f"({free_mb} MB free)"
                        )
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info_lines.append(f"Backend:       MPS (Apple Silicon)")
            else:
                info_lines.append(f"Backend:       CPU only")
                try:
                    cpu_count = os.cpu_count()
                    info_lines.append(f"CPU Cores:     {cpu_count}")
                except Exception:
                    pass

        # Fixed line spacing instead of np.linspace
        line_height = 0.08
        y_start = 0.95

        for i, line in enumerate(info_lines):
            y_pos = y_start - i * line_height
            if y_pos < 0.0:
                break  # don't draw lines that would go off the bottom
            ax.text(0.05, y_pos, line,
                    fontsize=7, fontfamily="monospace",
                    color="black",
                    transform=ax.transAxes,
                    verticalalignment="center")


    def update_predictions(self, predictions: list):
        """
        Update the predictions panel.

        Args:
            predictions: List of tuples (expected_str, predicted_str, is_correct)
        """
        if not self.enabled:
            return
        self._last_predictions = predictions
        self._draw_predictions()
        self._refresh()


    def set_model_info(self, info: dict):
        """Set model info to display."""
        self._model_info = info
        self._draw_model_info()
        self._refresh()


    def _draw_persistence_landscapes(self, landscapes_per_layer, sample_range, n_landscapes, resolution):
        """Draw overlaid persistence landscapes (one curve per layer) on ax_barcode."""
        ax = self.ax_barcode
        if ax is None:
            return

        ax.clear()
        ax.set_title(
                f"Persistence Landscapes (H₁) — step {self._topo_step}",
                fontsize=9, fontweight="bold",
                )

        if not landscapes_per_layer:
            ax.text(0.5, 0.5, "No H₁ features detected",
                    ha="center", va="center", fontsize=11, alpha=0.4,
                    transform=ax.transAxes)
            return

        n_layers = len(landscapes_per_layer)
        x_grid = np.linspace(sample_range[0], sample_range[1], resolution)

        # Use a colormap: early layers cool, late layers warm
        cmap = plt.cm.viridis
        colors = [cmap(i / max(n_layers - 1, 1)) for i in range(n_layers)]

        for li, ls_flat in enumerate(landscapes_per_layer):
            # ls_flat has shape (n_landscapes * resolution,)
            # Reshape to (n_landscapes, resolution) and plot only k=1 (the dominant landscape)
            ls_matrix = ls_flat.reshape(n_landscapes, resolution)
            dominant = ls_matrix[0]  # k=1 landscape

            if np.max(np.abs(dominant)) < 1e-12:
                continue  # skip flat-zero layers

            ax.plot(
                    x_grid, dominant,
                    color=colors[li],
                    linewidth=1.4,
                    alpha=0.7,
                    label=f"L{li}" if li % max(1, n_layers // 6) == 0 else None,
                    )

        ax.set_xlabel("Filtration value", fontsize=8)
        ax.set_ylabel("λ₁(t)", fontsize=8)
        ax.legend(fontsize=6, loc="upper right", ncol=2, framealpha=0.6)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)


    def _draw_wasserstein_heatmap(self, W_matrix):
        """Update the cross-layer Wasserstein-1 distance heatmap in-place."""
        ax = self.ax_bd
        if ax is None:
            return

        if W_matrix is None or W_matrix.size == 0:
            return

        # ── Normalize to [0, 1] ─────────────────────────────────────────
        w_max = W_matrix.max()
        if w_max > 0:
            W_normed = W_matrix / w_max
        else:
            W_normed = W_matrix

        # ── Update image data in-place (NO new imshow, NO new colorbar) ─
        if self._wass_im is not None:
            self._wass_im.set_data(W_normed)
            self._wass_im.set_extent((-0.5, W_normed.shape[1] - 0.5,
                                      -0.5, W_normed.shape[0] - 0.5))
        else:
            # Fallback: first call before _create_figure set it up
            self._wass_im = ax.imshow(
                    W_normed,
                    cmap="inferno",
                    interpolation="nearest",
                    origin="lower",
                    aspect="equal",
                    vmin=0.0,
                    vmax=1.0,
                    )

        # ── Restore axes position (undo any drift) ──────────────────────
        if self._wass_ax_pos is not None:
            ax.set_position(self._wass_ax_pos)

        # ── Update title and tick labels ────────────────────────────────
        ax.set_title(
                f"Wasserstein-1 Distance (layers) — step {self._topo_step}\n"
                f"[max={w_max:.2f}]",
                fontsize=9, fontweight="bold",
                )

        n = W_normed.shape[0]
        tick_positions = list(range(n))
        tick_labels = [f"L{i}" for i in range(n)]
        if n > 10:
            for i in range(len(tick_labels)):
                if i % 2 != 0:
                    tick_labels[i] = ""

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Layer", fontsize=8)


    # ── Finalize ────────────────────────────────────────────────────────
    def finalize(self):
        if not self.enabled:
            return

        # Always save final plot to file
        self._save_to_file()
        console.print(f"[bold green]📊 Final plot saved to: {self.plot_file}[/]")

        if not self.suppress_window:
            self.plt.ioff()
            self.plt.show()

# ════════════════════════════════════════════════════════════════════════════
# 8.  TIME ESTIMATOR
# ════════════════════════════════════════════════════════════════════════════

class TimeEstimator:
    def __init__(self):
        self.epoch_times: List[float] = []
        self.start_time = time.time()

    def add_epoch(self, elapsed: float):
        self.epoch_times.append(elapsed)

    @property
    def avg_epoch_time(self) -> float:
        if not self.epoch_times:
            return 0.0
        # Weighted average: recent epochs count more
        n = len(self.epoch_times)
        weights = [i + 1 for i in range(n)]
        total_w = sum(weights)
        return sum(t * w for t, w in zip(self.epoch_times, weights)) / total_w

    def eta(self, current_epoch: int, total_epochs: int) -> str:
        remaining = total_epochs - current_epoch
        if remaining <= 0:
            return "done"
        est = self.avg_epoch_time * remaining
        return self._fmt(est)

    def elapsed_total(self) -> str:
        return self._fmt(time.time() - self.start_time)

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m {s}s"


# ════════════════════════════════════════════════════════════════════════════
# 9.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# 9a. HELPER FUNCTIONS (extracted from train loop to eliminate duplication)
# ════════════════════════════════════════════════════════════════════════════

def _prepare_batch(
        tokenizer: BPETokenizer,
        batch_size: int,
        max_params: int,
        max_ops: int,
        allowed_ops: List[str],
        param_range: Tuple[int, int],
        max_seq_len: int,
        device: str,
        ) -> Optional[Tuple]:
    """
    Generate a batch, collate it, and move tensors to device.

    Returns:
        (batch_raw, inp, tgt, val_pos, val_tgt, ans_parseable)
        or None if the batch is too small.
    """
    batch = generate_batch(
            tokenizer, batch_size, max_params, max_ops,
            allowed_ops, param_range, max_seq_len,
            )
    if len(batch) < 2:
        return None

    inp, tgt, val_pos, val_tgt, ans_parseable = collate_batch(
            batch,
            pad_id=tokenizer.SPECIAL["<pad>"],
            tokenizer=tokenizer,
            )
    return (
            batch,
            inp.to(device),
            tgt.to(device),
            val_pos.to(device),
            val_tgt.to(device),
            ans_parseable.to(device),
            )


def _compute_batch_loss(
        model: TinyGPT,
        inp: torch.Tensor,
        tgt: torch.Tensor,
        val_pos: torch.Tensor,
        val_tgt: torch.Tensor,
        ans_parseable: torch.Tensor,
        tokenizer: BPETokenizer,
        device: str,
        value_loss_alpha: float,
        structure_loss_alpha: float,
        length_loss_alpha: float,
        ) -> Tuple[torch.Tensor, float, float, float, float]:
    """
    Thin wrapper around compute_structured_loss with standard args.

    Returns:
        (loss_tensor, ce_val, vloss_val, struct_val, parse_rate)
    """
    return compute_structured_loss(
            model, inp, tgt, val_pos, val_tgt, ans_parseable,
            tokenizer=tokenizer,
            device=device,
            alpha_value=value_loss_alpha,
            alpha_structure=structure_loss_alpha,
            alpha_length=length_loss_alpha,
            use_log_scale=True,
            )


def _save_checkpoint(
        path: str,
        filename: str,
        epoch: int,
        model: TinyGPT,
        optimizer,
        scheduler,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        ) -> str:
    """Save a .pt checkpoint to path/filename. Returns the full path."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        }, full_path)
    return full_path


def _prune_checkpoints(save_path: str, max_keep: int = 10):
    """Remove old model_epoch_*.pt files, keeping only the most recent max_keep."""
    import glob
    existing = sorted(
            glob.glob(os.path.join(save_path, "model_epoch_*.pt")),
            key=lambda p: int(
                os.path.basename(p).replace("model_epoch_", "").replace(".pt", "")
                ),
            )
    if len(existing) > max_keep:
        for old in existing[:-max_keep]:
            try:
                os.remove(old)
                console.print(
                        f"  [dim]🗑️  Removed old checkpoint: {os.path.basename(old)}[/]"
                        )
            except OSError:
                pass

def train(args: argparse.Namespace):
    allowed_ops = [op.strip() for op in args.allowed_ops.split(",")]
    valid_ops = list(list_supported_ops().keys())
    for op in allowed_ops:
        if op not in valid_ops:
            raise ValueError(f"Invalid op '{op}'. Valid: {valid_ops}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ── Plotter — open IMMEDIATELY ──────────────────────────────────────
    plotter = LivePlotter(
            enabled=not args.dont_plot,
            update_every=args.plot_every,
            topo_enabled=args.topo,
            topo_every=args.topo_every,
            topo_max_points=args.topo_max_points,
            topo_pca_dim=30,
            suppress_window=args.no_plot_window,
            plot_file=args.plot_file,
            )

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = build_tokenizer_from_samples(
            n_programs=args.tokenizer_initial_nr, allowed_ops=allowed_ops,
            max_params=args.max_params, max_ops=args.max_ops,
            param_range=(args.param_min, args.param_max),
            bpe_vocab_size=args.bpe_vocab_size,
            )

    # ── Model config ────────────────────────────────────────────────────
    if args.d_model > 0 and args.n_layers > 0 and args.n_heads > 0:
        cfg = {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                }
    else:
        console.print(f"[bold cyan]Searching for architecture "
                      f"(target ~{args.target_params:,} params)...[/]")
        cfg = find_model_config(tokenizer.vocab_size, args.target_params, args.max_seq_len)

    model_config = LLVMGPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            )

    # ── Create model ────────────────────────────────────────────────────
    model = TinyGPT(model_config).to(device)

    # ── Model summary in Rich panel ────────────────────────────────────
    from torchinfo import summary as torchinfo_summary

    model_stats = torchinfo_summary(
            model,
            input_size=(1, 128),
            dtypes=[torch.long],
            verbose=0,
            )

    summary_panel = Panel(
            Text(str(model_stats), style="white"),
            title="[bold cyan]📐 Model Summary (torchinfo)",
            border_style="cyan",
            padding=(1, 2),
            expand=False,
            )
    console.print(summary_panel)

    actual_params = model.count_parameters()

    plotter.set_model_info({
        "params": actual_params,
        "d_model": cfg["d_model"],
        "n_heads": cfg["n_heads"],
        "n_layers": cfg["n_layers"],
        "max_seq_len": args.max_seq_len,
        "vocab_size": tokenizer.vocab_size,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "lr": args.lr,
        "device": device,
        })

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 0
    resumed_train_losses = []
    resumed_val_losses = []
    resumed_best_val_loss = float("inf")
    resumed_total_samples = 0

    if args.resume is not None:
        if not os.path.isfile(args.resume):
            console.print(f"[bold red]❌ Checkpoint not found: {args.resume}[/]")
            sys.exit(1)

        console.print(f"[bold yellow]🔄 Resuming from checkpoint: {args.resume}[/]")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print("  [green]✓ Model weights loaded[/]")
        _resumed_checkpoint = checkpoint
    else:
        _resumed_checkpoint = None

    # ── Run logger ──────────────────────────────────────────────────────
    run_logger = None
    global run_dir
    if not args.no_run_log:
        run_logger = RunLogger(base_dir=args.run_dir)
        run_dir = run_logger.get_base_dir()

        console.print(f"[bold cyan]📁 Run logging to: {run_logger.path}[/]")
        run_logger.log_config(args)
        run_logger.log_model_summary(
                model_name="TinyGPT",
                param_count=actual_params,
                config_dict={
                    "d_model": cfg["d_model"],
                    "n_heads": cfg["n_heads"],
                    "n_layers": cfg["n_layers"],
                    "max_seq_len": args.max_seq_len,
                    "dropout": args.dropout,
                    },
                vocab_size=tokenizer.vocab_size,
                )

    # ── CSV Logger ──────────────────────────────────────────────────────
    global csv_log
    csv_log = CSVTrainingLogger(
            output_dir=run_dir if run_dir else "."
            )

    csv_log.set_output_dir(run_dir)

    # ── Config table ────────────────────────────────────────────────────
    config_table = Table(title="Configuration", box=box.ROUNDED, show_lines=True)
    config_table.add_column("Category", style="bold cyan")
    config_table.add_column("Parameter", style="bold")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model", "d_model", str(cfg["d_model"]))
    config_table.add_row("", "n_heads", str(cfg["n_heads"]))
    config_table.add_row("", "n_layers", str(cfg["n_layers"]))
    config_table.add_row("", "vocab_size", str(tokenizer.vocab_size))
    config_table.add_row("", "max_seq_len", str(args.max_seq_len))
    config_table.add_row("", "dropout", str(args.dropout))
    config_table.add_row("", "parameters", f"[bold]{actual_params:,}[/]")
    config_table.add_row("Training", "optimizer", args.optimizer)
    config_table.add_row("", "scheduler", args.scheduler)
    config_table.add_row("", "lr", str(args.lr))
    config_table.add_row("", "weight_decay", str(args.weight_decay))
    config_table.add_row("", "batch_size", str(args.batch_size))
    config_table.add_row("", "epochs", str(args.epochs))
    config_table.add_row("", "batches_per_epoch", str(args.batches_per_epoch))
    config_table.add_row("", "val_batches", str(args.val_batches))
    config_table.add_row("", "grad_clip", str(args.grad_clip))
    config_table.add_row("Data", "allowed_ops", ", ".join(allowed_ops))
    config_table.add_row("", "param_range", f"[{args.param_min}, {args.param_max}]")
    config_table.add_row("", "max_params", str(args.max_params))
    config_table.add_row("", "max_ops", str(args.max_ops))
    config_table.add_row("Infra", "device", device)
    config_table.add_row("", "seed", str(args.seed))
    config_table.add_row("", "plot_every", str(args.plot_every))
    config_table.add_row("Controls", "+ / =", "Add 10 epochs")
    config_table.add_row("", "-", "Remove 10 epochs")
    config_table.add_row("", "q", "Finish after current epoch")
    config_table.add_row("Controls", "r", "Reopen closed plot window")

    console.print(config_table)

    # ── Optimizer ───────────────────────────────────────────────────────
    opt_cls = OPTIMIZERS[args.optimizer]
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        opt_kwargs["momentum"] = args.momentum
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    if args.scheduler == "warmup_cosine":
        scheduler = build_warmup_cosine(optimizer, args.epochs, warmup_epochs=min(5, args.epochs // 5))
    else:
        scheduler = SCHEDULERS[args.scheduler](optimizer, args.epochs)
    is_plateau = args.scheduler == "plateau"

    # ── Epoch controller (keyboard) ─────────────────────────────────────
    epoch_ctrl = EpochController(initial_epochs=args.epochs, step=10, plotter=plotter)
    epoch_ctrl.start()

    # ── Time estimator ──────────────────────────────────────────────────
    timer = TimeEstimator()

    # ── Shared batch generation kwargs ──────────────────────────────────
    batch_gen_kwargs = dict(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_params=args.max_params,
            max_ops=args.max_ops,
            allowed_ops=allowed_ops,
            param_range=(args.param_min, args.param_max),
            max_seq_len=args.max_seq_len,
            device=device,
            )

    # ── Shared loss kwargs ──────────────────────────────────────────────
    loss_kwargs = dict(
            tokenizer=tokenizer,
            device=device,
            value_loss_alpha=args.value_loss_alpha,
            structure_loss_alpha=args.structure_loss_alpha,
            length_loss_alpha=args.length_loss_alpha,
            )

    # ── Training ────────────────────────────────────────────────────────
    console.print(
            Panel(
                f"[bold white]Training for {args.epochs} epochs  │  "
                f"{args.batches_per_epoch} batches/epoch  │  "
                f"batch_size={args.batch_size}  │  "
                f"Press +/- to adjust epochs, q to stop[/]",
                title="[bold green]🚀 Starting Training",
                border_style="green",
                )
            )

    best_val_loss = resumed_best_val_loss
    train_losses_hist: List[float] = list(resumed_train_losses)
    val_losses_hist: List[float] = list(resumed_val_losses)
    total_samples = resumed_total_samples
    save_path = f"{run_dir}/"

    epoch = start_epoch
    while True:
        epoch += 1

        total_epochs = epoch_ctrl.epochs
        if epoch > total_epochs:
            break
        epoch_ctrl.set_min(epoch)

        t0 = time.time()

        # ── Train epoch ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_value_loss = 0.0
        n_batches = 0

        plotter.clear_predictions()

        with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]Epoch {epoch}/{total_epochs}[/] Train"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[cyan]loss={task.fields[loss]:.4f}[/]"),
                TextColumn("•"),
                TextColumn("[dim]ema={task.fields[ema]:.4f}[/]"),
                TextColumn("•"),
                TextColumn("[dim]vloss={task.fields[vloss]:.4f}[/]"),
                TextColumn("•"),
                TextColumn("[dim]parse={task.fields[parse]:.0%}[/]"),
                TextColumn("•"),
                TextColumn("[dim]eta={task.fields[eta]}[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
                ) as progress:
            task = progress.add_task(
                    "train", total=args.batches_per_epoch,
                    loss=0.0, ema=0.0, vloss=0.0, parse=0.0, eta="...",
                    )

            ema_loss = None

            for batch_idx in range(args.batches_per_epoch):
                csv_log.start_batch_timer()

                prepared = _prepare_batch(**batch_gen_kwargs)
                if prepared is None:
                    progress.update(task, advance=1, loss=0.0,
                                    ema=ema_loss or 0.0, vloss=0.0,
                                    eta=timer.eta(epoch - 1, epoch_ctrl.epochs))
                    continue

                batch, inp, tgt, val_pos, val_tgt, ans_parseable = prepared
                total_samples += len(batch)

                # ── Combined loss ───────────────────────────────────────
                loss, ce_val, vloss_val, struct_val, parse_rate = _compute_batch_loss(
                        model, inp, tgt, val_pos, val_tgt, ans_parseable,
                        **loss_kwargs,
                        )

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip
                        )
                optimizer.step()

                bl = loss.item()
                epoch_loss += bl
                epoch_value_loss += vloss_val
                n_batches += 1

                if ema_loss is None:
                    ema_loss = bl
                else:
                    ema_loss = 0.05 * bl + 0.95 * ema_loss

                current_lr = optimizer.param_groups[0]["lr"]

                # ── Get predictions for this batch ──────────────────────
                preds = get_batch_predictions(model, tokenizer, batch, device)

                # ── CSV: log train batch ────────────────────────────────
                csv_log.log_train_batch(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        total_loss=bl,
                        ce_loss=ce_val,
                        value_loss=vloss_val,
                        structure_loss=struct_val,
                        parse_rate=parse_rate,
                        predictions=preds,
                        lr=current_lr,
                        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                        n_samples_in_batch=len(batch),
                        )

                # ── Plotter updates ─────────────────────────────────────
                plotter.update_batch(bl)
                plotter.update_topo(model, inp)
                plotter.update_kelp_forest(model, inp)

                if preds:
                    plotter.accumulate_predictions(preds)
                    plotter.update_prediction_diffs(preds)

                # ── Run logger ──────────────────────────────────────────
                if run_logger:
                    run_logger.log_batch_loss_train(epoch, batch_idx, bl, ema_loss)

                progress.update(
                        task, advance=1, loss=bl, ema=ema_loss,
                        vloss=vloss_val, parse=parse_rate,
                        eta=timer.eta(epoch - 1, epoch_ctrl.epochs),
                        )

        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_value_loss = epoch_value_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold blue]Epoch {epoch}/{total_epochs}[/] Val  "),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[magenta]loss={task.fields[loss]:.4f}[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
                ) as progress:
            task = progress.add_task("val", total=args.val_batches, loss=0.0)

            with torch.no_grad():
                for val_batch_idx in range(args.val_batches):
                    csv_log.start_batch_timer()

                    prepared = _prepare_batch(**batch_gen_kwargs)
                    if prepared is None:
                        progress.update(task, advance=1, loss=0.0)
                        continue

                    batch, inp, tgt, val_pos, val_tgt, ans_parseable = prepared

                    vl_total, vl_ce, vl_value, vl_struct, vl_parse_rate = _compute_batch_loss(
                            model, inp, tgt, val_pos, val_tgt, ans_parseable,
                            **loss_kwargs,
                            )

                    vl = vl_total.item()
                    val_loss += vl
                    val_batches += 1

                    # ── Get predictions for val batch ───────────────────
                    preds = get_batch_predictions(model, tokenizer, batch, device)

                    # ── CSV: log val batch ──────────────────────────────
                    csv_log.log_val_batch(
                            epoch=epoch,
                            batch_idx=val_batch_idx,
                            total_loss=vl,
                            ce_loss=vl_ce,
                            value_loss=vl_value,
                            structure_loss=vl_struct,
                            parse_rate=vl_parse_rate,
                            predictions=preds,
                            lr=optimizer.param_groups[0]["lr"],
                            )

                    # ── Plotter + run logger ────────────────────────────
                    plotter.update_val_batch(vl)

                    if val_batch_idx == 0 or val_batch_idx == args.val_batches - 1:
                        preds = get_batch_predictions(model, tokenizer, batch, device)
                        if preds:
                            plotter.update_predictions(preds)  # replaces, doesn't accumulate
                            plotter.update_prediction_diffs(preds)

                    if run_logger:
                        run_logger.log_batch_loss_val(epoch, val_batches, vl)

                    progress.update(task, advance=1, loss=vl)

        avg_val_loss = val_loss / max(val_batches, 1)

        plotter.finish_val_epoch()

        # ── Scheduler step ──────────────────────────────────────────────
        if is_plateau:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        timer.add_epoch(elapsed)

        # ── Track best ──────────────────────────────────────────────────
        train_losses_hist.append(avg_train_loss)
        val_losses_hist.append(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # ── CSV: end epoch ──────────────────────────────────────────────
        csv_log.end_epoch(
                epoch=epoch,
                lr=current_lr,
                epoch_time_sec=elapsed,
                )

        # ── Epoch summary ───────────────────────────────────────────────
        total_epochs = epoch_ctrl.epochs
        eta_str = timer.eta(epoch, total_epochs)
        elapsed_str = timer.elapsed_total()
        best_marker = " [bold green]★ best[/]" if is_best else ""

        epoch_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        for _ in range(12):
            epoch_table.add_column()

        epoch_table.add_row(
                "[bold]train:[/]", f"[cyan]{avg_train_loss:.4f}[/]",
                "[bold]val:[/]", f"[magenta]{avg_val_loss:.4f}[/]",
                "[bold]lr:[/]", f"[green]{current_lr:.2e}[/]",
                "[bold]samples:[/]", f"[yellow]{total_samples:,}[/]",
                "[bold]time:[/]", f"{elapsed:.1f}s",
                "[bold]eta:[/]", f"[dim]{eta_str}[/]{best_marker}",
                )

        console.print(
                Panel(
                    epoch_table,
                    title=f"[bold]Epoch {epoch}/{total_epochs}  │  elapsed: {elapsed_str}[/]",
                    border_style="green" if is_best else "blue",
                    width=min(console.width, 120),
                    )
                )

        # ── Run Logger: epoch + samples ─────────────────────────────────
        if run_logger:
            run_logger.log_epoch(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    lr=current_lr,
                    elapsed_secs=elapsed,
                    total_samples=total_samples,
                    is_best=is_best,
                    )

            n_to_log = None if args.log_all_samples else args.log_samples
            example_samples = generate_example_samples(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    num_samples=n_to_log if n_to_log else 999999,
                    max_params=args.max_params,
                    max_ops=args.max_ops,
                    allowed_ops=allowed_ops,
                    param_range=(args.param_min, args.param_max),
                    )
            run_logger.log_samples(epoch, example_samples, n_samples=n_to_log)
            run_logger.flush_losses()

        # ── Update epoch plot ───────────────────────────────────────────
        plotter.update_epoch(avg_train_loss, avg_val_loss, current_lr)

        # ── Checkpoint: save every epoch as model_epoch_N.pt ────────────
        ckpt_path = _save_checkpoint(
                path=save_path,
                filename=f"model_epoch_{epoch}.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                best_val_loss=best_val_loss,
                )
        console.print(f"  [dim]💾 Saved checkpoint: {ckpt_path}[/]")

        # ── Clean up old checkpoints: keep only the last 10 ────────────
        _prune_checkpoints(save_path, max_keep=10)

        # ── Save best model (lowest val loss) ───────────────────────────
        if is_best and save_path:
            best_path = _save_checkpoint(
                    path=save_path,
                    filename="model_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    best_val_loss=best_val_loss,
                    )
            # Also save in HuggingFace format for easy loading
            best_hf_path = f"{save_path}_best"
            model.save_pretrained(best_hf_path)
            tokenizer.save_pretrained(best_hf_path)
            console.print(
                    f"  [bold green]⭐ New best model saved: {best_path} "
                    f"(val_loss={avg_val_loss:.4f})[/]"
                    )

        if _interrupt_count >= 1:
            console.print("[bold yellow]Graceful stop after epoch.[/]")
            break

    # ── Stop controller ─────────────────────────────────────────────────
    epoch_ctrl.stop()

    # ── Close CSV logger ────────────────────────────────────────────────
    csv_log.close()
    console.print(
            f"[bold green]📊 CSV logs saved to: "
            f"{os.path.join(run_dir if run_dir else '.', 'batch_log.csv')} "
            f"and epoch_log.csv[/]"
            )

    # ── Save final model ────────────────────────────────────────────────
    if save_path:
        console.print(f"\n[bold cyan]Saving final model to {save_path} ...[/]")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        meta = {
                "train_losses": train_losses_hist,
                "val_losses": val_losses_hist,
                "best_val_loss": best_val_loss,
                "total_samples": total_samples,
                "actual_params": actual_params,
                "total_epochs": epoch,
                "elapsed": timer.elapsed_total(),
                "args": vars(args),
                }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        console.print(f"[green]✓ Saved: {save_path}/[/]")
        console.print("[dim]  config.json  pytorch_model.bin  tokenizer.json  training_meta.json[/]")

# ════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def compute_cross_layer_wasserstein(hidden_states, max_points=150, pca_dim=20):
    """Compute persistence diagrams per layer, then pairwise Wasserstein-1."""
    from ripser import ripser
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from persim import wasserstein as wasserstein_dist

    dgms_per_layer = []
    for hs in hidden_states:
        pts = hs.reshape(-1, hs.shape[-1]).cpu().float().numpy()
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]
        if pts.shape[1] > pca_dim:
            n_comp = min(pca_dim, pts.shape[0], pts.shape[1])
            pts = PCA(n_components=n_comp).fit_transform(
                    StandardScaler().fit_transform(pts)
                    )

        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        thresh = np.percentile(dists[dists > 0], 50)
        result = ripser(pts, maxdim=1, thresh=thresh)
        dgms_per_layer.append(result['dgms'])

    n = len(dgms_per_layer)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            total_dist = 0.0
            for dim in range(min(len(dgms_per_layer[i]), len(dgms_per_layer[j]))):
                d_i = dgms_per_layer[i][dim]
                d_j = dgms_per_layer[j][dim]
                d_i = d_i[np.isfinite(d_i[:, 1])]
                d_j = d_j[np.isfinite(d_j[:, 1])]
                if len(d_i) > 0 and len(d_j) > 0:
                    total_dist += wasserstein_dist(d_i, d_j)
            W[i, j] = W[j, i] = total_dist
    return W

def compute_jacobian_spectra(model, input_ids, device):
    """Compute singular values of the per-layer Jacobian via finite differences."""
    model.eval()
    output = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states

    spectra = []
    for ell in range(len(hidden_states) - 1):
        h_in = hidden_states[ell][0].detach()    # (T, D)
        h_out = hidden_states[ell + 1][0].detach()  # (T, D)
        delta = h_out - h_in  # residual delta

        # Estimate local Jacobian via SVD of the delta cloud
        # Center both
        h_in_c = h_in - h_in.mean(0)
        delta_c = delta - delta.mean(0)

        # Least-squares Jacobian: delta ≈ h_in @ J^T
        # J = (h_in^T h_in)^{-1} h_in^T delta
        try:
            J = torch.linalg.lstsq(h_in_c, delta_c).solution  # (D, D)
            svs = torch.linalg.svdvals(J).cpu().numpy()
        except:
            svs = np.zeros(h_in.shape[-1])
        spectra.append(svs)

    return spectra

def compute_jacobian_invariants(spectra_or_jacobians):
    """From per-layer Jacobians, extract div/curl/shear per token."""
    results = []
    for J in jacobians:  # J: (D, D) per layer
        div = torch.trace(J).item()
        J_antisym = (J - J.T) / 2
        curl = torch.norm(J_antisym, p='fro').item()
        J_sym = (J + J.T) / 2
        shear_mat = J_sym - (torch.trace(J_sym) / J.shape[0]) * torch.eye(J.shape[0], device=J.device)
        shear = torch.norm(shear_mat, p='fro').item()
        results.append({'div': div, 'curl': curl, 'shear': shear})
    return results

def compute_cka_matrix(hidden_states):
    """Layer-to-layer CKA similarity."""
    def linear_cka(X, Y):
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        hsic_xy = torch.norm(X.T @ Y, p='fro') ** 2
        hsic_xx = torch.norm(X.T @ X, p='fro') ** 2
        hsic_yy = torch.norm(Y.T @ Y, p='fro') ** 2
        return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-8)).item()

    n = len(hidden_states)
    cka = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            X = hidden_states[i].reshape(-1, hidden_states[i].shape[-1]).float()
            Y = hidden_states[j].reshape(-1, hidden_states[j].shape[-1]).float()
            cka[i, j] = cka[j, i] = linear_cka(X, Y)
    return cka

def compute_persistence_landscapes(hidden_states, n_landscapes=5, resolution=100,
                                   max_points=150, pca_dim=20):
    """
    Compute persistence landscapes per layer for cross-layer comparison.

    Args:
        hidden_states: tuple of tensors (B, T, D), one per layer
        n_landscapes:  number of landscape functions (k=1..n_landscapes)
        resolution:    number of sample points along the filtration axis
        max_points:    subsample limit per layer
        pca_dim:       PCA target dimension

    Returns:
        landscapes_per_layer: list of (n_landscapes * resolution,) arrays, one per layer
        sample_range:         (min_birth, max_death) across all layers for consistent x-axis
    """
    from ripser import ripser
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # ── Pass 1: compute all persistence diagrams and find global range ──
    all_dgms_h1 = []
    global_min = float('inf')
    global_max = float('-inf')

    for hs in hidden_states:
        pts = hs.reshape(-1, hs.shape[-1]).cpu().float().numpy()

        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        if pts.shape[1] > pca_dim:
            pts = PCA(n_components=min(pca_dim, pts.shape[0], pts.shape[1])).fit_transform(
                StandardScaler().fit_transform(pts)
            )

        dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        thresh = np.percentile(dists[dists > 0], 50)

        result = ripser(pts, maxdim=1, thresh=thresh)
        dgm_h1 = result['dgms'][1]  # H1 = loops
        finite = dgm_h1[np.isfinite(dgm_h1[:, 1])]

        if len(finite) > 0:
            global_min = min(global_min, finite[:, 0].min())
            global_max = max(global_max, finite[:, 1].max())

        all_dgms_h1.append(finite)

    # Handle edge case: no finite H1 features anywhere
    if global_min >= global_max:
        global_min, global_max = 0.0, 1.0

    sample_range = (global_min, global_max)

    # ── Pass 2: compute landscapes on a SHARED grid ─────────────────────
    landscape_fn = _Landscape(
        num_landscapes=n_landscapes,
        resolution=resolution,
        sample_range=sample_range,
    )

    landscapes_per_layer = []
    for finite in all_dgms_h1:
        if len(finite) >= 2:
            # gudhi expects a list of diagrams
            ls = landscape_fn.fit_transform([finite])  # (1, n_landscapes * resolution)
            landscapes_per_layer.append(ls[0])
        else:
            # No features → flat zero landscape
            landscapes_per_layer.append(np.zeros(n_landscapes * resolution))

    return landscapes_per_layer, sample_range

if __name__ == "__main__":
    console.print(
        Panel(
            "[bold white]LLVM IR GPT Trainer[/]\n"
            "[dim]Learning to execute LLVM IR[/]",
            border_style="bold cyan",
            padding=(1, 4),
        )
    )

    # ── Wait for another process to finish before claiming the GPU ──
    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid)

    try:
        model, tokenizer = train(args)
    except KeyboardInterrupt:
        console.print("\n[bold red]Training interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/]")
        console.print_exception()

