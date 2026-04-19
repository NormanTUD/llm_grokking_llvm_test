# train_llvm_gpt.py

"""
Train a tiny GPT-like transformer to predict the integer output of
randomly generated LLVM IR functions.

Dependencies:
    pip install torch matplotlib llvmlite rich

Usage:
    python train_llvm_gpt.py --target-params 100000 --epochs 30 --lr 3e-4
    python train_llvm_gpt.py --help
"""

import argparse
import math
import random
import time
import threading
from typing import List, Tuple, Optional

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
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

from random_llvm_gen import generate_random_function, list_supported_ops

console = Console()


# ════════════════════════════════════════════════════════════════════════════
# 1.  TOKENIZER
# ════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<sep>": 3}

    def __init__(self):
        self.char2idx: dict[str, int] = {}
        self.idx2char: dict[int, str] = {}
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


# ════════════════════════════════════════════════════════════════════════════
# 2.  DYNAMIC DATA GENERATION
# ════════════════════════════════════════════════════════════════════════════

def generate_single_sample(
    tokenizer: CharTokenizer,
    max_params: int = 4,
    max_ops: int = 6,
    allowed_ops: Optional[List[str]] = None,
    param_range: Tuple[int, int] = (-50, 50),
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
    return ids


class DynamicDataGenerator:
    """
    Generates training/validation data on-the-fly in a background thread.
    Maintains a buffer of ready-to-use samples.
    """

    def __init__(
        self,
        tokenizer: CharTokenizer,
        buffer_size: int = 1000,
        max_params: int = 3,
        max_ops: int = 4,
        allowed_ops: Optional[List[str]] = None,
        param_range: Tuple[int, int] = (-20, 20),
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.max_params = max_params
        self.max_ops = max_ops
        self.allowed_ops = allowed_ops or ["add", "sub", "mul"]
        self.param_range = param_range
        self.max_seq_len = max_seq_len

        self._buffer: List[List[int]] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._generated_count = 0
        self._failed_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._fill_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _fill_loop(self):
        while self._running:
            with self._lock:
                current_size = len(self._buffer)
            if current_size >= self.buffer_size:
                time.sleep(0.01)
                continue

            sample = generate_single_sample(
                self.tokenizer,
                self.max_params,
                self.max_ops,
                self.allowed_ops,
                self.param_range,
            )
            if sample is not None and len(sample) <= self.max_seq_len:
                with self._lock:
                    self._buffer.append(sample)
                    self._generated_count += 1
            else:
                self._failed_count += 1

    def get_batch(self, batch_size: int) -> List[List[int]]:
        """Pull a batch from the buffer. Blocks briefly if buffer is low."""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if len(self._buffer) >= batch_size:
                    batch = self._buffer[:batch_size]
                    self._buffer = self._buffer[batch_size:]
                    return batch
            time.sleep(0.01)
        # Return whatever we have
        with self._lock:
            batch = self._buffer[:]
            self._buffer.clear()
        return batch

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "total_generated": self._generated_count,
                "total_failed": self._failed_count,
            }


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
# 3.  TINY GPT MODEL
# ════════════════════════════════════════════════════════════════════════════

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
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
    target_params: int = 100_000,
    max_seq_len: int = 2048,
) -> dict:
    best = None
    best_diff = float("inf")
    for d_model in range(16, 512, 4):
        for n_heads in (1, 2, 4, 8):
            if d_model % n_heads != 0:
                continue
            for n_layers in range(1, 16):
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
    "warmup_cosine": None,  # handled specially below
}


def build_warmup_cosine(optimizer, epochs, warmup_epochs=5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════════
# 6.  MATPLOTLIB PLOTTER (non-blocking)
# ════════════════════════════════════════════════════════════════════════════

class LivePlotter:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        self.plt = plt

        self.plt.ion()
        self.fig, self.axes = self.plt.subplots(2, 2, figsize=(16, 10))
        self.fig.suptitle("LLVM IR GPT Training", fontsize=14, fontweight="bold")
        self.fig.canvas.manager.set_window_title("LLVM IR GPT — Live Training Monitor")

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.lr_history: List[float] = []
        self.batch_losses: List[float] = []
        self.buffer_sizes: List[int] = []

    def update_batch(self, batch_loss: float):
        if not self.enabled:
            return
        self.batch_losses.append(batch_loss)

    def update_epoch(
        self,
        train_loss: float,
        val_loss: float,
        lr: float,
        buffer_size: int,
    ):
        if not self.enabled:
            return

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_history.append(lr)
        self.buffer_sizes.append(buffer_size)

        ax_loss, ax_batch, ax_lr, ax_buf = self.axes.flat

        # Epoch losses
        ax_loss.clear()
        ax_loss.plot(self.train_losses, label="Train", color="steelblue", linewidth=2)
        ax_loss.plot(self.val_losses, label="Val", color="tomato", linewidth=2)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Epoch Loss (Cross-Entropy)")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Batch losses (smoothed)
        ax_batch.clear()
        if len(self.batch_losses) > 0:
            # Exponential moving average
            alpha = 0.05
            smoothed = []
            val = self.batch_losses[0]
            for bl in self.batch_losses:
                val = alpha * bl + (1 - alpha) * val
                smoothed.append(val)
            ax_batch.plot(smoothed, color="mediumpurple", linewidth=1, alpha=0.9)
            ax_batch.plot(self.batch_losses, color="mediumpurple", linewidth=0.3, alpha=0.3)
        ax_batch.set_xlabel("Batch (global)")
        ax_batch.set_ylabel("Loss")
        ax_batch.set_title("Batch Loss (EMA smoothed)")
        ax_batch.grid(True, alpha=0.3)

        # Learning rate
        ax_lr.clear()
        ax_lr.plot(self.lr_history, color="seagreen", linewidth=2)
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("LR")
        ax_lr.set_title("Learning Rate")
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(-3, -3))

        # Buffer utilization
        ax_buf.clear()
        ax_buf.plot(self.buffer_sizes, color="darkorange", linewidth=2, marker="o", markersize=3)
        ax_buf.set_xlabel("Epoch")
        ax_buf.set_ylabel("Samples")
        ax_buf.set_title("Data Buffer Size")
        ax_buf.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.plt.pause(0.01)

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

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = CharTokenizer()

    console.print("\n[bold cyan]Building vocabulary from pilot samples...[/]")
    for _ in range(200):
        generate_single_sample(tokenizer, args.max_params, args.max_ops, allowed_ops,
                               (args.param_min, args.param_max))

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

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)

    actual_params = model.count_parameters()

    # ── Print config table ──────────────────────────────────────────────
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
    config_table.add_row("", "parameters", f"{actual_params:,}")
    config_table.add_row("Training", "optimizer", args.optimizer)
    config_table.add_row("", "scheduler", args.scheduler)
    config_table.add_row("", "lr", str(args.lr))
    config_table.add_row("", "weight_decay", str(args.weight_decay))
    config_table.add_row("", "batch_size", str(args.batch_size))
    config_table.add_row("", "epochs", str(args.epochs))
    config_table.add_row("", "grad_clip", str(args.grad_clip))
    config_table.add_row("Data", "batches_per_epoch", str(args.batches_per_epoch))
    config_table.add_row("", "val_batches", str(args.val_batches))
    config_table.add_row("", "allowed_ops", ", ".join(allowed_ops))
    config_table.add_row("", "param_range", f"[{args.param_min}, {args.param_max}]")
    config_table.add_row("", "max_params", str(args.max_params))
    config_table.add_row("", "max_ops", str(args.max_ops))
    config_table.add_row("Infra", "device", device)
    config_table.add_row("", "seed", str(args.seed))

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

    # ── Dynamic data generators ─────────────────────────────────────────
    console.print("\n[bold cyan]Starting dynamic data generators...[/]")

    train_gen = DynamicDataGenerator(
        tokenizer=tokenizer,
        buffer_size=args.batch_size * 20,
        max_params=args.max_params,
        max_ops=args.max_ops,
        allowed_ops=allowed_ops,
        param_range=(args.param_min, args.param_max),
        max_seq_len=args.max_seq_len,
    )
    val_gen = DynamicDataGenerator(
        tokenizer=tokenizer,
        buffer_size=args.batch_size * 10,
        max_params=args.max_params,
        max_ops=args.max_ops,
        allowed_ops=allowed_ops,
        param_range=(args.param_min, args.param_max),
        max_seq_len=args.max_seq_len,
    )

    train_gen.start()
    val_gen.start()

    # Wait for initial buffer fill
    console.print("[dim]Waiting for data buffers to fill...[/]")
    while train_gen.stats["buffer_size"] < args.batch_size * 2:
        time.sleep(0.1)
    console.print(f"[green]✓ Train buffer ready "
                  f"({train_gen.stats['buffer_size']} samples)[/]")

    while val_gen.stats["buffer_size"] < args.batch_size * 2:
        time.sleep(0.1)
    console.print(f"[green]✓ Val buffer ready "
                  f"({val_gen.stats['buffer_size']} samples)[/]")

    # ── Plotter ─────────────────────────────────────────────────────────
    plotter = LivePlotter(enabled=args.plot)

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
            TextColumn("[dim]buf={task.fields[buf]}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "train",
                total=args.batches_per_epoch,
                loss=0.0,
                buf=0,
            )

            for batch_idx in range(args.batches_per_epoch):
                batch = train_gen.get_batch(args.batch_size)
                if len(batch) < 2:
                    time.sleep(0.05)
                    continue

                inp, tgt = collate_batch(batch, pad_id=tokenizer.SPECIAL["<pad>"])
                inp, tgt = inp.to(device), tgt.to(device)

                _, loss = model(inp, tgt)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                n_batches += 1

                plotter.update_batch(batch_loss)

                progress.update(
                    task,
                    advance=1,
                    loss=batch_loss,
                    buf=train_gen.stats["buffer_size"],
                )

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────────
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
            task = progress.add_task(
                "val",
                total=args.val_batches,
                loss=0.0,
            )

            with torch.no_grad():
                for _ in range(args.val_batches):
                    batch = val_gen.get_batch(args.batch_size)
                    if len(batch) < 2:
                        time.sleep(0.05)
                        continue

                    inp, tgt = collate_batch(batch, pad_id=tokenizer.SPECIAL["<pad>"])
                    inp, tgt = inp.to(device), tgt.to(device)

                    _, loss = model(inp, tgt)
                    val_loss += loss.item()
                    val_batches += 1

                    progress.update(task, advance=1, loss=loss.item())

        avg_val_loss = val_loss / max(val_batches, 1)

        # ── Scheduler step ──────────────────────────────────────────
        if is_plateau:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # ── Track best ──────────────────────────────────────────────
        train_losses_hist.append(avg_train_loss)
        val_losses_hist.append(avg_val_loss)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        # ── Epoch summary ───────────────────────────────────────────
        gen_stats = train_gen.stats
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
            "buf:", f"{gen_stats['buffer_size']}/{gen_stats['total_generated']}",
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

        # ── Update plot ─────────────────────────────────────────────
        plotter.update_epoch(
            avg_train_loss,
            avg_val_loss,
            current_lr,
            gen_stats["buffer_size"],
        )

        # ── Checkpoint ──────────────────────────────────────────────
        if args.save_every > 0 and epoch % args.save_every == 0:
            path = f"llvm_gpt_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "config": cfg,
                "args": vars(args),
            }, path)
            console.print(f"  [dim]💾 Saved checkpoint: {path}[/]")

        if is_best and args.save_path:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_char2idx": tokenizer.char2idx,
                "args": vars(args),
                "train_losses": train_losses_hist,
                "val_losses": val_losses_hist,
            }, args.save_path)

    # ── Stop generators ─────────────────────────────────────────────────
    train_gen.stop()
    val_gen.stop()

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
    summary_table.add_row("Total Samples Generated",
                          f"{train_gen.stats['total_generated'] + val_gen.stats['total_generated']:,}")
    summary_table.add_row("Total Failed Generations",
                          f"{train_gen.stats['total_failed'] + val_gen.stats['total_failed']:,}")
    if args.save_path:
        summary_table.add_row("Best Model Saved To", args.save_path)

    console.print()
    console.print(summary_table)

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
    g.add_argument("--target-params", type=int, default=100_000,
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
    g.add_argument("--save-every", type=int, default=0,
                   help="Save checkpoint every N epochs (0 = disabled)")
    g.add_argument("--save-path", type=str, default="llvm_gpt_final.pt",
                   help="Path to save the best model")

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
