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

run_dir = None

import argparse
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
import random
from generate_samples import generate_example_samples
from random_llvm_gen import generate_random_function
import subprocess
from run_logger import RunLogger
from generate_samples import generate_example_samples
import threading
import tty
import termios
import select
import matplotlib
import matplotlib.pyplot as plt
import tkinter

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
        except ProcessNotFoundError:
            break
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
import os as _os

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


def collate_batch(batch, pad_id=0):
    max_len = max(len(s) for s, _ in batch)
    input_ids, target_ids = [], []
    for s, prompt_len in batch:
        padded = s + [pad_id] * (max_len - len(s))
        inp = padded[:-1]
        tgt = padded[1:]
        # Mask out everything before the answer
        for i in range(min(prompt_len - 1, len(tgt))):
            tgt[i] = pad_id  # ignored by cross_entropy with ignore_index=0
        input_ids.append(inp)
        target_ids.append(tgt)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

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
        output_hidden_states: bool = False,
        **kwargs,
    ):
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

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0,
            )

        return _ModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(hidden_states_list) if output_hidden_states else None,
            last_hidden_state=x,
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


class _ModelOutput(dict):
    def __init__(self, loss=None, logits=None, hidden_states=None, last_hidden_state=None):
        super().__init__(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            last_hidden_state=last_hidden_state,
        )
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state

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
                 model_info: dict = None):
        self.enabled = enabled
        self._model_info = model_info or {}
        self._last_predictions = []  # Store last batch predictions
        self.update_every = update_every
        self._global_batch = 0
        self._ema_train = None
        self.suppress_window = suppress_window
        self.plot_file = plot_file
        self._window_closed = False
        self._reopen_requested = False
        self._lock = threading.Lock()
        self._abs_diffs_history: List[List[int]] = []  # per-update list of abs diffs

        # ── TDA config ──────────────────────────────────────────────────
        self.topo_enabled = topo_enabled and _HAS_RIPSER and enabled
        self.topo_every = topo_every
        self.topo_max_points = topo_max_points
        self.topo_pca_dim = topo_pca_dim
        self._topo_step = 0
        self._topo_dgms = None
        self._topo_layer_name = ""

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
            # Start a thread that watches for window close events
            self._window_watch_thread = threading.Thread(
                target=self._watch_window, daemon=True
            )
            self._window_watch_thread.start()

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
            # 3 rows × 3 cols: top row = epoch/batch/lr,
            #                   mid row = val/barcode/birth-death,
            #                   bot row = predictions (wide) + model info
            self.fig = self.plt.figure(figsize=(24, 16))
            gs = self.fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

            ax_epoch   = self.fig.add_subplot(gs[0, 0])
            ax_batch   = self.fig.add_subplot(gs[0, 1])
            ax_lr      = self.fig.add_subplot(gs[0, 2])
            ax_val     = self.fig.add_subplot(gs[1, 0])
            ax_barcode = self.fig.add_subplot(gs[1, 1])
            ax_bd      = self.fig.add_subplot(gs[1, 2])
            ax_preds   = self.fig.add_subplot(gs[2, 0:2])   # spans 2 columns
            ax_info    = self.fig.add_subplot(gs[2, 2])

            self.ax_barcode = ax_barcode
            self.ax_bd = ax_bd
            self.ax_diffs = None
        else:
            self.fig = self.plt.figure(figsize=(22, 14))
            gs = self.fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

            ax_epoch = self.fig.add_subplot(gs[0, 0])
            ax_batch = self.fig.add_subplot(gs[0, 1])
            ax_lr    = self.fig.add_subplot(gs[0, 2])
            ax_val   = self.fig.add_subplot(gs[1, 0])
            ax_diffs = self.fig.add_subplot(gs[1, 1])
            ax_preds = self.fig.add_subplot(gs[1, 2])
            ax_info  = self.fig.add_subplot(gs[2, :])

            ax_diffs.set_title("Absolute Prediction Error (outliers removed)",
                               fontsize=10, fontweight="bold")
            ax_diffs.set_xlabel("Prediction Update #")
            ax_diffs.set_ylabel("|expected - predicted|")
            ax_diffs.grid(True, alpha=0.3)
            self.line_diffs_mean, = ax_diffs.plot(
                [], [], label="Mean |diff|", color="darkorange", linewidth=2,
            )
            self.line_diffs_median, = ax_diffs.plot(
                [], [], label="Median |diff|", color="purple", linewidth=1.5,
                linestyle="--",
            )
            ax_diffs.legend(loc="upper right", fontsize=8)
            self.ax_diffs = ax_diffs

            self.ax_barcode = None
            self.ax_bd = None

        # ── REMOVED the unconditional overwrite that was here ──
        # These two lines were killing the topo axes even when topo was enabled:
        #   self.ax_barcode = None
        #   self.ax_bd = None

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

        # ── Window title ────────────────────────────────────────────────
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("LLVM IR GPT — Live Training")
        except Exception:
            pass

        # ── Top-left: Epoch losses ──────────────────────────────────────
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

        # ── Top-center: Batch loss EMA ──────────────────────────────────
        ax_batch.set_title("Batch Loss (Train EMA)", fontsize=10, fontweight="bold")
        ax_batch.set_xlabel("Batch")
        ax_batch.set_ylabel("Loss")
        ax_batch.grid(True, alpha=0.3)
        self.line_batch_ema, = ax_batch.plot(
            [], [], label="Train EMA", color="steelblue", linewidth=2,
        )
        ax_batch.legend(loc="upper right", fontsize=8)

        # ── Top-right: Learning Rate ────────────────────────────────────
        ax_lr.set_title("Learning Rate", fontsize=10, fontweight="bold")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
        self.line_lr, = ax_lr.plot(
            [], [], label="LR", color="seagreen", linewidth=2,
        )
        ax_lr.legend(loc="upper right", fontsize=8)

        # ── Mid-left: Val batch loss ────────────────────────────────────
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

        # ── TDA panels (only when topo_enabled) ─────────────────────────
        if self.ax_barcode is not None:
            self.ax_barcode.set_title("TDA Barcode (Embedding Space)",
                                       fontsize=10, fontweight="bold")
            self.ax_barcode.text(
                0.5, 0.5, "Waiting for data...",
                ha="center", va="center",
                transform=self.ax_barcode.transAxes,
                fontsize=11, alpha=0.4,
            )
            self.ax_barcode.grid(True, alpha=0.2, axis="x")

        if self.ax_bd is not None:
            self.ax_bd.set_title("TDA Birth / Death (Persistence)",
                                  fontsize=10, fontweight="bold")
            self.ax_bd.text(
                0.5, 0.5, "Waiting for data...",
                ha="center", va="center",
                transform=self.ax_bd.transAxes,
                fontsize=11, alpha=0.4,
            )
            self.ax_bd.grid(True, alpha=0.2)

        # ── Predictions panel (text only) ───────────────────────────────
        ax_preds.set_xlim(0, 1)
        ax_preds.set_ylim(0, 1)
        ax_preds.axis("off")
        ax_preds.set_title(
            "Last Batch Predictions (Expected → Got)",
            fontsize=10, fontweight="bold",
        )
        self._draw_predictions()

        # ── Model Info panel (text only) ────────────────────────────────
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        ax_info.set_title("Model / System Info", fontsize=10, fontweight="bold")
        self._draw_model_info()

        # ── Layout ──────────────────────────────────────────────────────
        self.fig.tight_layout()

        # ── Restore existing data onto the new figure ───────────────────
        self._restore_data()

        # ── Register close event ────────────────────────────────────────
        if not self.suppress_window:
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        # ── Force window to appear ──────────────────────────────────────
        if not self.suppress_window:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def _restore_data(self):
        """Re-plot all accumulated data onto current line objects."""
        if self.train_epoch_losses:
            epochs = list(range(1, len(self.train_epoch_losses) + 1))
            self.line_train_epoch.set_data(epochs, self.train_epoch_losses)
            self.line_val_epoch.set_data(epochs, self.val_epoch_losses)

        if self.batch_ema:
            xs = list(range(len(self.batch_ema)))
            self.line_batch_ema.set_data(xs, self.batch_ema)

        if self.lr_history:
            epochs = list(range(1, len(self.lr_history) + 1))
            self.line_lr.set_data(epochs, self.lr_history)

        if self.val_batch_raw:
            vxs = list(range(len(self.val_batch_raw)))
            self.line_val_raw.set_data(vxs, self.val_batch_raw)
            self.line_val_epoch_avg.set_data(self._val_epoch_avg_xs,
                                              self._val_epoch_avg_ys)

        # Redraw TDA if we have cached diagrams
        if self._topo_dgms is not None and self.ax_barcode is not None:
            self._draw_barcode(self._topo_dgms, self._topo_layer_name)
            self._draw_birth_death(self._topo_dgms, self._topo_layer_name)

        self._draw_predictions()
        self._draw_model_info()

        # Restore diff plot
        if hasattr(self, 'ax_diffs') and self.ax_diffs is not None and self._abs_diffs_history:
            means = [np.mean(d) for d in self._abs_diffs_history]
            medians = [np.median(d) for d in self._abs_diffs_history]
            xs = list(range(len(means)))
            self.line_diffs_mean.set_data(xs, means)
            self.line_diffs_median.set_data(xs, medians)

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
            # Only relim/autoscale on actual plot axes, not text panels
            for ax in self._plot_axes:
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
            if hidden_states is None:
                return

            n_layers = len(hidden_states)
            if n_layers >= 3:
                layer_idx = n_layers // 2
                layer_name = f"Block {layer_idx - 1}"
            else:
                layer_idx = n_layers - 1
                layer_name = "Embed+Pos" if layer_idx == 0 else f"Block {layer_idx - 1}"

            hs = hidden_states[layer_idx]
            points = hs.reshape(-1, hs.shape[-1]).cpu().float().numpy()

            if points.shape[0] > self.topo_max_points:
                idx = np.random.choice(points.shape[0],
                                       self.topo_max_points, replace=False)
                points = points[idx]

            if _HAS_SKLEARN and points.shape[1] > self.topo_pca_dim:
                scaler = _StandardScaler()
                points = scaler.fit_transform(points)
                n_comp = min(self.topo_pca_dim, points.shape[0], points.shape[1])
                pca = _PCA(n_components=n_comp)
                points = pca.fit_transform(points)

            if points.shape[0] > 100:
                sample = points[np.random.choice(points.shape[0], 100, replace=False)]
            else:
                sample = points

            dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=-1)
            thresh = np.percentile(dists[dists > 0], 50)

            result = _ripser_fn(points, maxdim=1, thresh=thresh)
            dgms = result["dgms"]

            self._topo_dgms = dgms
            self._topo_layer_name = layer_name

            self._draw_barcode(dgms, layer_name)
            self._draw_birth_death(dgms, layer_name)

            self._refresh()

        except Exception:
            pass
        finally:
            if was_training:
                model.train()

    def update_prediction_diffs(self, predictions: list):
        """
        Collect absolute differences from predictions, remove NaN/extreme outliers,
        and update the diff plot.
        """
        if not self.enabled:
            return

        # 1) Collect diffs from this batch
        diffs = []
        for expected, predicted, _ in predictions:
            try:
                exp_val = int(expected.strip())
                pred_val = int(predicted.strip())
                diffs.append(abs(exp_val - pred_val))
            except (ValueError, TypeError):
                continue

        if not diffs:
            return

        # 2) Remove extreme outliers (beyond 95th percentile)
        diffs_arr = np.array(diffs, dtype=float)
        if len(diffs_arr) > 2:
            p95 = np.percentile(diffs_arr, 95)
            diffs_arr = diffs_arr[diffs_arr <= p95]

        # Remove any remaining NaN/inf
        diffs_arr = diffs_arr[np.isfinite(diffs_arr)]

        if len(diffs_arr) == 0:
            return

        # 3) Append FIRST, then plot
        self._abs_diffs_history.append(diffs_arr.tolist())

        # 4) Update the diff plot lines
        if hasattr(self, 'ax_diffs') and self.ax_diffs is not None:
            means = [np.mean(d) for d in self._abs_diffs_history]
            medians = [np.median(d) for d in self._abs_diffs_history]
            xs = list(range(len(means)))
            self.line_diffs_mean.set_data(xs, means)
            self.line_diffs_median.set_data(xs, medians)
            self.ax_diffs.relim()
            self.ax_diffs.autoscale_view()
            self._refresh()


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

        n_show = min(len(self._last_predictions), 16)
        y_positions = np.linspace(0.95, 0.05, n_show)

        for i, (expected, predicted, is_correct) in enumerate(
            self._last_predictions[-n_show:]
        ):
            color = "green" if is_correct else "red"
            marker = "✓" if is_correct else "✗"

            try:
                exp_val = int(expected.strip())
                pred_val = int(predicted.strip())
                diff = abs(exp_val - pred_val)
                text = (
                    f"{marker}  expected: {exp_val:>6d}  │  "
                    f"got: {pred_val:>6d}  │  diff: {diff}"
                )
            except (ValueError, TypeError):
                text = (
                    f"{marker}  expected: {expected[:12]:>12s}  │  "
                    f"got: {predicted[:12]:>12s}"
                )

            ax.text(0.02, y_positions[i], text,
                    fontsize=8, fontfamily="monospace",
                    color=color, transform=ax.transAxes,
                    verticalalignment="center")

        n_correct = sum(1 for _, _, c in self._last_predictions if c)
        n_total = len(self._last_predictions)
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0
        ax.text(0.98, 0.01,
                f"Accuracy: {n_correct}/{n_total} ({accuracy:.1f}%)",
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
            f"Architecture:  TinyGPT",
            f"Parameters:    {self._model_info.get('params', '?'):,}",
            f"d_model:       {self._model_info.get('d_model', '?')}",
            f"n_heads:       {self._model_info.get('n_heads', '?')}",
            f"n_layers:      {self._model_info.get('n_layers', '?')}",
            f"max_seq_len:   {self._model_info.get('max_seq_len', '?')}",
            f"vocab_size:    {self._model_info.get('vocab_size', '?')}",
            f"dropout:       {self._model_info.get('dropout', '?')}",
            f"optimizer:     {self._model_info.get('optimizer', '?')}",
            f"scheduler:     {self._model_info.get('scheduler', '?')}",
            f"lr:            {self._model_info.get('lr', '?')}",
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
            info_lines.append(f"Platform:      {platform.system()} {platform.release()}")
            info_lines.append(f"Python:        {platform.python_version()}")
            info_lines.append(f"PyTorch:       {torch.__version__}")
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

        y_positions = np.linspace(0.95, 0.02, len(info_lines))
        for i, line in enumerate(info_lines):
            ax.text(0.05, y_positions[i], line,
                    fontsize=8, fontfamily="monospace",
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


    def _draw_barcode(self, dgms: list, layer_name: str):
        ax = self.ax_barcode
        if ax is None:
            return

        ax.clear()
        ax.set_title(f"TDA Barcode — {layer_name}  (step {self._topo_step})",
                      fontsize=9, fontweight="bold")

        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        dim_labels = ["H₀ (components)", "H₁ (loops)", "H₂ (voids)"]

        y_offset = 0
        y_ticks = []
        y_tick_labels = []

        for dim, dgm in enumerate(dgms):
            if dim > 1:
                break

            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                continue

            persistences = finite[:, 1] - finite[:, 0]
            order = np.argsort(-persistences)
            finite = finite[order]

            start_y = y_offset
            for bar in finite:
                birth, death = bar
                ax.plot(
                    [birth, death], [y_offset, y_offset],
                    color=colors[dim % len(colors)],
                    linewidth=2.0, alpha=0.8,
                    solid_capstyle="butt",
                )
                y_offset += 1

            mid_y = (start_y + y_offset) / 2
            y_ticks.append(mid_y)
            y_tick_labels.append(
                dim_labels[dim] if dim < len(dim_labels) else f"H_{dim}"
            )

        ax.set_xlabel("Filtration value", fontsize=8)
        if y_ticks:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=8)
        else:
            ax.set_yticks([])
        ax.grid(True, alpha=0.2, axis="x")
        ax.tick_params(axis="x", labelsize=7)

    def _draw_birth_death(self, dgms: list, layer_name: str):
        ax = self.ax_bd
        if ax is None:
            return

        ax.clear()
        ax.set_title(f"TDA Birth/Death — {layer_name}  (step {self._topo_step})",
                      fontsize=9, fontweight="bold")

        colors = ["#2196F3", "#FF5722", "#4CAF50"]
        labels = ["H₀", "H₁", "H₂"]

        all_vals = []
        for dim, dgm in enumerate(dgms):
            if dim > 1:
                break
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) == 0:
                continue

            all_vals.extend(finite.flatten().tolist())
            ax.scatter(
                finite[:, 0], finite[:, 1],
                s=25, alpha=0.7,
                color=colors[dim % len(colors)],
                label=labels[dim] if dim < len(labels) else f"H_{dim}",
                edgecolors="white", linewidth=0.4,
                zorder=3,
            )

        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.1 + 1e-6
            ax.plot(
                [lo - margin, hi + margin],
                [lo - margin, hi + margin],
                "k--", alpha=0.3, linewidth=1, zorder=1,
            )
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

        ax.set_xlabel("Birth", fontsize=8)
        ax.set_ylabel("Death", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="box")

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
        enabled=args.plot,
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
        verbose=0,  # suppress direct printing
    )

    # Build a Rich table from the summary string
    summary_text = str(model_stats)
    summary_panel = Panel(
        Text(summary_text, style="white"),
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

        # We'll load optimizer & scheduler state AFTER they're created below
        _resumed_checkpoint = checkpoint
    else:
        _resumed_checkpoint = None

    run_logger = None
    if not args.no_run_log:
        run_logger = RunLogger(base_dir=args.run_dir)
        global run_dir
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
        n_batches = 0

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
            TextColumn("[dim]eta={task.fields[eta]}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "train", total=args.batches_per_epoch, loss=0.0, ema=0.0, eta="..."
            )
            ema_loss = None

            for batch_idx in range(args.batches_per_epoch):
                batch = generate_batch(
                    tokenizer, args.batch_size, args.max_params, args.max_ops,
                    allowed_ops, (args.param_min, args.param_max), args.max_seq_len,
                )
                total_samples += len(batch)

                if len(batch) < 2:
                    progress.update(task, advance=1, loss=0.0, ema=ema_loss or 0.0,
                                    eta=timer.eta(epoch - 1, epoch_ctrl.epochs))
                    continue

                inp, tgt = collate_batch(batch, pad_id=tokenizer.SPECIAL["<pad>"])
                inp, tgt = inp.to(device), tgt.to(device)

                output = model(input_ids=inp, labels=tgt)
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                bl = loss.item()
                epoch_loss += bl
                n_batches += 1

                if ema_loss is None:
                    ema_loss = bl
                else:
                    ema_loss = 0.05 * bl + 0.95 * ema_loss

                plotter.update_batch(bl)
                plotter.update_topo(model, inp)

                if run_logger:
                    run_logger.log_batch_loss_train(epoch, batch_idx, bl, ema_loss)

                if batch:
                    preds = get_batch_predictions(model, tokenizer, batch, device)
                    plotter.update_predictions(preds)
                    plotter.update_prediction_diffs(preds)

                progress.update(
                    task, advance=1, loss=bl, ema=ema_loss,
                    eta=timer.eta(epoch - 1, epoch_ctrl.epochs),
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

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
                for _ in range(args.val_batches):
                    batch = generate_batch(
                        tokenizer, args.batch_size, args.max_params, args.max_ops,
                        allowed_ops, (args.param_min, args.param_max), args.max_seq_len,
                    )
                    total_samples += len(batch)

                    if len(batch) < 2:
                        progress.update(task, advance=1, loss=0.0)
                        continue

                    inp, tgt = collate_batch(batch, pad_id=tokenizer.SPECIAL["<pad>"])
                    inp, tgt = inp.to(device), tgt.to(device)

                    output = model(input_ids=inp, labels=tgt)
                    vl = output.loss.item()
                    val_loss += vl
                    val_batches += 1

                    plotter.update_val_batch(vl)
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

        save_path = f"{run_dir}/"

        # ── Checkpoint: save every epoch as model_epoch_N.pt ────────────
        epoch_ckpt_path = os.path.join(save_path, f"model_epoch_{epoch}.pt")
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
        }, epoch_ckpt_path)
        console.print(f"  [dim]💾 Saved checkpoint: {epoch_ckpt_path}[/]")

        # ── Save best model (lowest val loss) ───────────────────────────
        if is_best and save_path:
            best_path = os.path.join(save_path, "model_best.pt")
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            }, best_path)
            # Also save in HuggingFace format for easy loading
            best_hf_path = f"{save_path}_best"
            model.save_pretrained(best_hf_path)
            tokenizer.save_pretrained(best_hf_path)
            console.print(f"  [bold green]⭐ New best model saved: {best_path} (val_loss={avg_val_loss:.4f})[/]")

        if _interrupt_count >= 1:
            console.print("[bold yellow]Graceful stop after epoch.[/]")
            break


    # ── Stop controller ─────────────────────────────────────────────────
    epoch_ctrl.stop()

    # ── Save final model ────────────────────────────────────────────────
    if save_path:
        console.print(f"\n[bold cyan]Saving final model to {save_path}/ ...[/]")
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

    # ── Final summary ───────────────────────────────────────────────────
    summary_table = Table(
        title="Training Complete",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        title_style="bold green",
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Final Train Loss", f"{train_losses_hist[-1]:.4f}")
    summary_table.add_row("Final Val Loss", f"{val_losses_hist[-1]:.4f}")
    summary_table.add_row("Best Val Loss", f"{best_val_loss:.4f}")
    summary_table.add_row("Model Parameters", f"{actual_params:,}")
    summary_table.add_row("Total Samples", f"{total_samples:,}")
    summary_table.add_row("Total Epochs", str(epoch))
    summary_table.add_row("Total Time", timer.elapsed_total())
    summary_table.add_row("Avg Epoch Time", f"{timer.avg_epoch_time:.1f}s")

    save_path = f"{run_dir}/"

    summary_table.add_row("Save Path", save_path)

    console.print()
    console.print(summary_table)

    if save_path:
        console.print(
            Panel(
                f'[bold]from train_llvm_gpt import TinyGPT, BPETokenizer\n\n'
                f'model = TinyGPT.from_pretrained("{save_path}")\n'
                f'model.eval()\n\n'
                f'tokenizer = BPETokenizer.from_pretrained("{save_path}")\n'
                f'config = model.config[/]',
                title="[bold green]📦 Load your model",
                border_style="green",
            )
        )

    console.print("Finalizing plotters")
    plotter.finalize()

    if run_logger:
        run_logger.log_final_summary(
            train_losses=train_losses_hist,
            val_losses=val_losses_hist,
            best_val_loss=best_val_loss,
            total_samples=total_samples,
            total_epochs=epoch,
            total_time=timer.elapsed_total(),
            param_count=actual_params,
            save_path=save_path,
        )
        console.print(f"[bold green]📁 Run data saved to: {run_logger.path}[/]")

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════════
# 10.  ARGPARSE
# ════════════════════════════════════════════════════════════════════════════

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
    g.add_argument("--plot", action="store_true", default=True,
                   help="Enable live matplotlib")
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
    g.add_argument("--topo-every", type=int, default=50,
                   help="Update topological barcodes every N batches")
    g.add_argument("--topo-max-points", type=int, default=200,
                   help="Max points to subsample for persistence computation")

    return p.parse_args()

# ════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

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

