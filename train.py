# train_llvm_gpt.py

"""
Train a tiny GPT-like transformer to predict the integer output of
randomly generated LLVM IR functions.

Dependencies:
    pip install torch matplotlib llvmlite rich transformers

Usage:
    python train_llvm_gpt.py --target-params 1000 --epochs 30 --lr 3e-4
    python train_llvm_gpt.py --help
"""

import argparse
import json
import math
import os
import random
import time
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

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

console = Console()


# ════════════════════════════════════════════════════════════════════════════
# 1.  TOKENIZER  (HuggingFace-compatible save/load)
# ════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3}

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self._next_id = len(self.SPECIAL)
        seed_chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " \n\t=@%{}(),.*:;!<>_-+/\"#"
        )
        for ch in seed_chars:
            self._register(ch)

    def _register(self, ch: str) -> int:
        if ch not in self.char2idx:
            idx = self._next_id
            self.char2idx[ch] = idx
            self.idx2char[idx] = ch
            self._next_id += 1
        return self.char2idx[ch]

    @property
    def vocab_size(self) -> int:
        return self._next_id

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.SPECIAL["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL["<eos>"]

    def encode(self, text: str) -> List[int]:
        return [self._register(ch) for ch in text]

    def decode(self, ids: List[int]) -> str:
        inv_special = {v: k for k, v in self.SPECIAL.items()}
        parts = []
        for i in ids:
            if i in inv_special:
                parts.append(inv_special[i])
            elif i in self.idx2char:
                parts.append(self.idx2char[i])
            else:
                parts.append("?")
        return "".join(parts)

    def __call__(self, text: str, return_tensors: str = None, **kwargs) -> dict:
        """HuggingFace-compatible __call__."""
        ids = (
            [self.SPECIAL["<bos>"]]
            + self.encode(text)
            + [self.SPECIAL["<eos>"]]
        )
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
            "next_id": self._next_id,
            "special": self.SPECIAL,
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f, indent=2)
        # Also write a tokenizer_config.json for HF compatibility
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump({"tokenizer_class": "CharTokenizer"}, f)

    @classmethod
    def from_pretrained(cls, path: str) -> "CharTokenizer":
        tok = cls.__new__(cls)
        tok.char2idx = {}
        tok.idx2char = {}
        with open(os.path.join(path, "tokenizer.json"), "r") as f:
            data = json.load(f)
        tok.char2idx = data["char2idx"]
        tok.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        tok._next_id = data["next_id"]
        return tok


# ════════════════════════════════════════════════════════════════════════════
# 2.  DATA GENERATION (per-batch, on the fly)
# ════════════════════════════════════════════════════════════════════════════

def generate_single_sample(
    tokenizer: CharTokenizer,
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
        [tokenizer.SPECIAL["<bos>"]]
        + tokenizer.encode(text)
        + [tokenizer.SPECIAL["<eos>"]]
    )
    if len(ids) > max_seq_len:
        return None
    return ids


def generate_batch(
    tokenizer: CharTokenizer,
    batch_size: int,
    max_params: int = 3,
    max_ops: int = 4,
    allowed_ops: Optional[List[str]] = None,
    param_range: Tuple[int, int] = (-20, 20),
    max_seq_len: int = 2048,
) -> List[List[int]]:
    """Generate a fresh random batch. Retries on failure."""
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


def collate_batch(
    batch: List[List[int]], pad_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in batch)
    input_ids = []
    target_ids = []
    for s in batch:
        padded = s + [pad_id] * (max_len - len(s))
        input_ids.append(padded[:-1])
        target_ids.append(padded[1:])
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
        target_ids, dtype=torch.long
    )


# ════════════════════════════════════════════════════════════════════════════
# 3.  TINY GPT MODEL  (HuggingFace-compatible save/load)
# ════════════════════════════════════════════════════════════════════════════

class LLVMGPTConfig:
    """Minimal config object compatible with HF's config pattern."""
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
        self.hidden_size = d_model  # HF compat
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
    """
    Small GPT-style decoder-only transformer with HuggingFace-compatible
    save_pretrained / from_pretrained.
    """

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
        # Accept and ignore extra HF kwargs
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

        for block in self.blocks:
            x = block(x)
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

        # Return a simple namespace that mimics HF output
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
        # Allow overriding config values
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


class _ModelOutput:
    """Minimal HuggingFace-style model output."""
    def __init__(self, loss=None, logits=None, hidden_states=None, last_hidden_state=None):
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
# 6.  LIVE PLOTTER — opens immediately, updates per batch
# ════════════════════════════════════════════════════════════════════════════

class LivePlotter:
    """
    Non-blocking matplotlib plotter.
    Opens the window immediately and updates every `update_every` batches.
    Uses draw_idle() + flush_events() for responsive updates.
    """

    def __init__(self, enabled: bool = True, update_every: int = 10):
        self.enabled = enabled
        self.update_every = update_every
        self._global_batch = 0

        if not enabled:
            return

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt

        self.plt.ion()
        self.fig, self.axes = self.plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        try:
            self.fig.canvas.manager.set_window_title("LLVM IR GPT — Live Training")
        except Exception:
            pass

        # Initialize empty line objects for efficient updates
        ax_epoch, ax_batch, ax_lr, ax_val = self.axes.flat

        ax_epoch.set_title("Epoch Loss")
        ax_epoch.set_xlabel("Epoch")
        ax_epoch.set_ylabel("Loss")
        ax_epoch.grid(True, alpha=0.3)
        self.line_train_epoch, = ax_epoch.plot([], [], label="Train", color="steelblue", linewidth=2)
        self.line_val_epoch, = ax_epoch.plot([], [], label="Val", color="tomato", linewidth=2)
        ax_epoch.legend()

        ax_batch.set_title("Batch Loss (EMA)")
        ax_batch.set_xlabel("Batch (global)")
        ax_batch.set_ylabel("Loss")
        ax_batch.grid(True, alpha=0.3)
        self.line_batch_raw, = ax_batch.plot([], [], color="mediumpurple", linewidth=0.3, alpha=0.3)
        self.line_batch_ema, = ax_batch.plot([], [], color="mediumpurple", linewidth=2, alpha=0.9)

        ax_lr.set_title("Learning Rate")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))
        self.line_lr, = ax_lr.plot([], [], color="seagreen", linewidth=2)

        ax_val.set_title("Validation Loss per Batch")
        ax_val.set_xlabel("Val Batch (global)")
        ax_val.set_ylabel("Loss")
        ax_val.grid(True, alpha=0.3)
        self.line_val_batch, = ax_val.plot([], [], color="tomato", linewidth=1, alpha=0.7)
        self.line_val_ema, = ax_val.plot([], [], color="tomato", linewidth=2)

        self.fig.tight_layout()

        # Data storage
        self.train_epoch_losses: List[float] = []
        self.val_epoch_losses: List[float] = []
        self.batch_losses: List[float] = []
        self.batch_ema: List[float] = []
        self.lr_history: List[float] = []
        self.val_batch_losses: List[float] = []
        self.val_batch_ema: List[float] = []
        self._ema_val = None
        self._ema_train = None

        # Force initial draw so window appears immediately
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.01)

    def _refresh(self):
        """Efficiently redraw only changed artists."""
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_batch(self, batch_loss: float):
        if not self.enabled:
            return

        self._global_batch += 1
        self.batch_losses.append(batch_loss)

        # EMA
        alpha = 0.05
        if self._ema_train is None:
            self._ema_train = batch_loss
        else:
            self._ema_train = alpha * batch_loss + (1 - alpha) * self._ema_train
        self.batch_ema.append(self._ema_train)

        if self._global_batch % self.update_every == 0:
            xs = list(range(len(self.batch_losses)))
            self.line_batch_raw.set_data(xs, self.batch_losses)
            self.line_batch_ema.set_data(xs, self.batch_ema)
            self._refresh()

    def update_val_batch(self, val_batch_loss: float):
        if not self.enabled:
            return

        self.val_batch_losses.append(val_batch_loss)
        alpha = 0.1
        if self._ema_val is None:
            self._ema_val = val_batch_loss
        else:
            self._ema_val = alpha * val_batch_loss + (1 - alpha) * self._ema_val
        self.val_batch_ema.append(self._ema_val)

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

        vxs = list(range(len(self.val_batch_losses)))
        self.line_val_batch.set_data(vxs, self.val_batch_losses)
        self.line_val_ema.set_data(vxs, self.val_batch_ema)

        self._refresh()

    def finalize(self):
        if not self.enabled:
            return
        self.plt.ioff()
        self.plt.show()


# ════════════════════════════════════════════════════════════════════════════
# 7.  TRAINING LOOP
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
    plotter = LivePlotter(enabled=args.plot, update_every=args.plot_every)

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = CharTokenizer()
    console.print("[bold cyan]Building vocabulary from pilot samples...[/]")
    for _ in range(200):
        generate_single_sample(
            tokenizer, args.max_params, args.max_ops, allowed_ops,
            (args.param_min, args.param_max), args.max_seq_len,
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

    model = TinyGPT(model_config).to(device)
    actual_params = model.count_parameters()

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
    config_table.add_row("", "save_path", args.save_path)

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

    # ── Training ────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold white]Training for {args.epochs} epochs  │  "
            f"{args.batches_per_epoch} batches/epoch  │  "
            f"batch_size={args.batch_size}[/]",
            title="[bold green]🚀 Starting Training",
            border_style="green",
        )
    )

    best_val_loss = float("inf")
    train_losses_hist: List[float] = []
    val_losses_hist: List[float] = []
    total_samples_generated = 0
    total_samples_failed = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train epoch ─────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Epoch {epoch}/{args.epochs}[/] Train"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[cyan]loss={task.fields[loss]:.4f}[/]"),
            TextColumn("•"),
            TextColumn("[dim]ema={task.fields[ema]:.4f}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "train",
                total=args.batches_per_epoch,
                loss=0.0,
                ema=0.0,
            )

            ema_loss = None

            for batch_idx in range(args.batches_per_epoch):
                # Generate a fresh random batch on the fly
                batch = generate_batch(
                    tokenizer,
                    args.batch_size,
                    args.max_params,
                    args.max_ops,
                    allowed_ops,
                    (args.param_min, args.param_max),
                    args.max_seq_len,
                )
                total_samples_generated += len(batch)

                if len(batch) < 2:
                    total_samples_failed += args.batch_size
                    progress.update(task, advance=1, loss=0.0, ema=ema_loss or 0.0)
                    continue

                inp, tgt = collate_batch(batch, pad_id=tokenizer.SPECIAL["<pad>"])
                inp, tgt = inp.to(device), tgt.to(device)

                output = model(input_ids=inp, labels=tgt)
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1

                if ema_loss is None:
                    ema_loss = batch_loss
                else:
                    ema_loss = 0.05 * batch_loss + 0.95 * ema_loss

                plotter.update_batch(batch_loss)

                progress.update(
                    task,
                    advance=1,
                    loss=batch_loss,
                    ema=ema_loss,
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Epoch {epoch}/{args.epochs}[/] Val  "),
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
                        tokenizer,
                        args.batch_size,
                        args.max_params,
                        args.max_ops,
                        allowed_ops,
                        (args.param_min, args.param_max),
                        args.max_seq_len,
                    )
                    total_samples_generated += len(batch)

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
                    progress.update(task, advance=1, loss=vl)

        avg_val_loss = val_loss / max(val_batches, 1)

        # ── Scheduler step ──────────────────────────────────────────────
        if is_plateau:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ── Track best ──────────────────────────────────────────────────
        train_losses_hist.append(avg_train_loss)
        val_losses_hist.append(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # ── Epoch summary ───────────────────────────────────────────────
        best_marker = " [bold green]★ best[/]" if is_best else ""

        epoch_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        epoch_table.add_column(style="bold")
        epoch_table.add_column(style="cyan")
        epoch_table.add_column(style="bold")
        epoch_table.add_column(style="magenta")
        epoch_table.add_column(style="bold")
        epoch_table.add_column(style="green")
        epoch_table.add_column(style="bold")
        epoch_table.add_column(style="yellow")
        epoch_table.add_column(style="bold")
        epoch_table.add_column()

        epoch_table.add_row(
            "train:", f"{avg_train_loss:.4f}",
            "val:", f"{avg_val_loss:.4f}",
            "lr:", f"{current_lr:.2e}",
            "samples:", f"{total_samples_generated:,}",
            "time:", f"{elapsed:.1f}s{best_marker}",
        )

        console.print(
            Panel(
                epoch_table,
                title=f"[bold]Epoch {epoch}/{args.epochs}[/]",
                border_style="blue" if not is_best else "green",
                width=100,
            )
        )

        # ── Update epoch plot ───────────────────────────────────────────
        plotter.update_epoch(avg_train_loss, avg_val_loss, current_lr)

        # ── Checkpoint ──────────────────────────────────────────────────
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = f"{args.save_path}_epoch{epoch}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            console.print(f"  [dim]💾 Saved checkpoint: {ckpt_path}/[/]")

        if is_best and args.save_path:
            best_path = f"{args.save_path}_best"
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    # ── Save final model (HuggingFace-compatible) ───────────────────────
    if args.save_path:
        console.print(f"\n[bold cyan]Saving final model to {args.save_path}/ ...[/]")
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)

        # Also save training metadata
        meta = {
            "train_losses": train_losses_hist,
            "val_losses": val_losses_hist,
            "best_val_loss": best_val_loss,
            "total_samples_generated": total_samples_generated,
            "actual_params": actual_params,
            "args": vars(args),
        }
        with open(os.path.join(args.save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        console.print(f"[green]✓ Model saved to {args.save_path}/[/]")
        console.print(f"[dim]  Files: config.json, pytorch_model.bin, tokenizer.json, training_meta.json[/]")

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
    summary_table.add_row("Total Samples Generated", f"{total_samples_generated:,}")
    summary_table.add_row("Save Path", args.save_path or "(none)")

    console.print()
    console.print(summary_table)

    # ── Show how to load ────────────────────────────────────────────────
    if args.save_path:
        console.print(
            Panel(
                f'[bold]from train_llvm_gpt import TinyGPT, CharTokenizer\n\n'
                f'model = TinyGPT.from_pretrained("{args.save_path}")\n'
                f'model.eval()\n\n'
                f'tokenizer = CharTokenizer.from_pretrained("{args.save_path}")\n'
                f'config = model.config[/]',
                title="[bold green]📦 Load your model",
                border_style="green",
            )
        )

    plotter.finalize()

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════════
# 8.  ARGPARSE
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a tiny GPT on randomly generated LLVM IR functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data generation ─────────────────────────────────────────────────
    g = p.add_argument_group("Data Generation")
    g.add_argument("--batches-per-epoch", type=int, default=150,
                   help="Number of training batches per epoch")
    g.add_argument("--val-batches", type=int, default=20,
                   help="Number of validation batches per epoch")
    g.add_argument("--max-params", type=int, default=3,
                   help="Max number of function parameters")
    g.add_argument("--max-ops", type=int, default=4,
                   help="Max number of operations in the random DAG")
    g.add_argument("--allowed-ops", type=str, default="add,sub,mul",
                   help="Comma-separated list of allowed LLVM ops")
    g.add_argument("--param-min", type=int, default=-20,
                   help="Minimum value for random function parameters")
    g.add_argument("--param-max", type=int, default=20,
                   help="Maximum value for random function parameters")

    # ── Model architecture ──────────────────────────────────────────────
    g = p.add_argument_group("Model Architecture")
    g.add_argument("--target-params", type=int, default=1_000,
                   help="Target number of model parameters (used for auto config)")
    g.add_argument("--d-model", type=int, default=0,
                   help="Model dimension (0 = auto from target-params)")
    g.add_argument("--n-heads", type=int, default=0,
                   help="Number of attention heads (0 = auto)")
    g.add_argument("--n-layers", type=int, default=0,
                   help="Number of transformer layers (0 = auto)")
    g.add_argument("--max-seq-len", type=int, default=2048,
                   help="Maximum sequence length")
    g.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout rate")

    # ── Training ────────────────────────────────────────────────────────
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
                   help="Momentum (only for SGD)")
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping max norm")
    g.add_argument("--optimizer", type=str, default="adamw",
                   choices=list(OPTIMIZERS.keys()),
                   help="Optimizer to use")
    g.add_argument("--scheduler", type=str, default="cosine",
                   choices=list(SCHEDULERS.keys()),
                   help="LR scheduler to use")

    # ── Infrastructure ──────────────────────────────────────────────────
    g = p.add_argument_group("Infrastructure")
    g.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device to train on")
    g.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    g.add_argument("--plot", action="store_true", default=True,
                   help="Enable live matplotlib plotting")
    g.add_argument("--no-plot", action="store_false", dest="plot",
                   help="Disable live plotting")
    g.add_argument("--plot-every", type=int, default=5,
                   help="Update matplotlib plot every N batches")
    g.add_argument("--save-every", type=int, default=0,
                   help="Save checkpoint every N epochs (0 = disabled)")
    g.add_argument("--save-path", type=str, default="llvm_gpt_model",
                   help="Directory to save the model (HuggingFace format)")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 9.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    banner = Text()
    banner.append("╔══════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║   ", style="bold cyan")
    banner.append("LLVM IR GPT Trainer", style="bold white")
    banner.append("                   ║\n", style="bold cyan")
    banner.append("║   ", style="bold cyan")
    banner.append("Learning to execute LLVM IR", style="dim white")
    banner.append("          ║\n", style="bold cyan")
    banner.append("╚══════════════════════════════════════════╝", style="bold cyan")
    console.print(banner)

    try:
        model, tokenizer = train(args)
    except KeyboardInterrupt:
        console.print("\n[bold red]Training interrupted by user.[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/]")
        console.print_exception()
