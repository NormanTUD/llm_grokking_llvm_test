# train_llvm_gpt.py

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "matplotlib",
#   "llvmlite",
#   "ripser",
#   "persim",
#   "gudhi",
#   "scikit-learn",
#   "scipy",
#   "tokenizers",
#   "torchinfo",
#   "rich",
# ]
# ///

original_print = print

import os
import sys

from datetime import datetime, timedelta  # Add this line back!

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc

current_epoch = 0

def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

def should_set_exclude_newer():
    return not os.environ.get("UV_EXCLUDE_NEWER")


def restart_with_uv(script_path, args, env):
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def ensure_safe_env():
    if not should_set_exclude_newer():
        return
    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date
    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


# Must run BEFORE heavy imports
ensure_safe_env()

# ═══════════════════════════════════════════════════════
# Now all your existing imports and code follow...
# ═══════════════════════════════════════════════════════
run_dir = None

import json
import math

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

from random_infix_gen import generate_random_function, list_supported_ops
from turnstile_gen import generate_turnstile_function
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

import torch
import torch.nn as nn
import copy
import math
import numpy as np
from collections import deque
from typing import Optional, Tuple, List

_sync_proc: Optional[subprocess.Popen] = None
_sync_started: bool = False

def _maybe_run_sync(run_dir_path: Optional[str], sync_target: Optional[str]):
    """
    Launch watch_and_sync.sh ONCE in the background.
    It loops internally (while true + sleep), so we only ever start it once.
    If it died unexpectedly, restart it.
    """
    global _sync_proc, _sync_started

    if sync_target is None or run_dir_path is None:
        return

    # Already running? Nothing to do.
    if _sync_proc is not None and _sync_proc.poll() is None:
        return

    # If it was started before but died, log it
    if _sync_started and _sync_proc is not None:
        rc = _sync_proc.returncode
        stderr_out = ""
        try:
            stderr_out = _sync_proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            pass
        console.print(f"  [yellow]⚠ watch_and_sync.sh died (exit code {rc}) — restarting...[/]")
        if stderr_out:
            console.print(f"  [dim red]{stderr_out[-500:]}[/]")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sync_script = os.path.join(script_dir, "watch_and_sync.sh")

    if not os.path.isfile(sync_script):
        console.print(f"  [yellow]⚠ watch_and_sync.sh not found at {sync_script}[/]")
        return

    # Ensure trailing slash on run_dir for consistency with the script
    source = run_dir_path.rstrip("/") + "/"

    cmd = [
        "bash", sync_script,
        source,
        "--copy-to", sync_target,
    ]

    console.print(f"  [dim]🔄 Launching watch_and_sync.sh → {sync_target} (background, loops internally)[/]")
    try:
        _sync_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,  # detach from Ctrl+C
        )
        _sync_started = True
    except Exception as e:
        console.print(f"  [yellow]⚠ Failed to launch watch_and_sync.sh: {e}[/]")


def _wait_for_sync():
    """
    At the end of training, let watch_and_sync.sh do ONE final cycle,
    then kill it gracefully.
    """
    global _sync_proc

    if _sync_proc is None or _sync_proc.poll() is not None:
        return

    console.print("[dim]🔄 Letting watch_and_sync.sh finish one last sync cycle...[/]")

    # Give it time to complete one cycle (its internal INTERVAL + processing)
    try:
        _sync_proc.wait(timeout=120)
        console.print("[green]✓ watch_and_sync.sh exited on its own.[/]")
        return
    except subprocess.TimeoutExpired:
        pass

    # It's still looping (expected) — kill it gracefully
    console.print("[dim]🔄 Stopping watch_and_sync.sh...[/]")
    try:
        os.killpg(os.getpgid(_sync_proc.pid), signal.SIGTERM)
        _sync_proc.wait(timeout=15)
        console.print("[green]✓ watch_and_sync.sh stopped.[/]")
    except Exception:
        try:
            _sync_proc.kill()
            _sync_proc.wait(timeout=5)
            console.print("[yellow]⚠ watch_and_sync.sh force-killed.[/]")
        except Exception:
            console.print("[yellow]⚠ Could not stop watch_and_sync.sh — it may still be running.[/]")

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes cos/sin frequencies for all positions up to max_seq_len,
    then applies rotation to query/key pairs in attention.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the frequency bands: theta_i = base^(-2i/dim) for i in [0, dim//2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions [0, seq_len)."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # Outer product: (seq_len,) x (dim//2,) -> (seq_len, dim//2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate to cover full dim: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Register as buffers (not parameters, but move with .to(device))
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        """
        Returns (cos, sin) each of shape (seq_len, dim) for the given sequence length.
        """
        if seq_len > self.max_seq_len:
            # Dynamically extend cache if needed
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE rotation to a tensor x of shape (B, n_heads, T, head_dim).

    cos, sin: (T, head_dim) — precomputed from RotaryPositionEmbedding.

    The rotation formula for each pair (x_2i, x_{2i+1}):
        x_2i'    = x_2i * cos - x_{2i+1} * sin
        x_{2i+1}' = x_{2i+1} * cos + x_2i * sin

    Implemented efficiently by splitting into two halves and rotating.
    """
    # x: (B, n_heads, T, head_dim)
    # cos, sin: (T, head_dim) -> reshape to (1, 1, T, head_dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)

    # Rotate: split x into two halves along last dim, swap and negate
    x1 = x[..., : x.shape[-1] // 2]  # first half
    x2 = x[..., x.shape[-1] // 2 :]  # second half
    # "Rotated" version: [-x2, x1]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    return (x * cos) + (x_rotated * sin)

class AdaptiveLocalMinimaExplorer(torch.optim.lr_scheduler._LRScheduler):
    """
    A learning rate scheduler that:

    1. Detects when the optimizer has settled into a local minimum
       (loss plateau detection via moving average convergence)
    2. Saves the current "home base" (model state + optimizer state)
    3. Launches exploration probes: large LR → small LR cycles
       in different effective directions (via perturbation + momentum reset)
    4. If a probe finds a consistently better basin, migrates there
    5. If not, rolls back to home base and tries another direction

    The "directions" in high-dimensional space are implicitly controlled by:
      - Resetting optimizer momentum/state (changes effective gradient direction)
      - Adding small random perturbations to parameters before probing
      - Varying the exploration magnitude across probes

    Conceptual model:
      - Current state = point in R^N (N = number of parameters)
      - Local minimum = region where loss is flat/oscillating
      - Exploration = temporarily increase LR to escape, then anneal
      - Evaluation = track whether the new trajectory consistently improves
      - Rollback = restore saved state if exploration failed
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        # ── Plateau detection ───────────────────────────────────
        patience: int = 20,              # batches of flat loss before declaring plateau
        plateau_threshold: float = 1e-4, # relative improvement threshold
        loss_window: int = 50,           # window for moving average

        # ── Exploration parameters ──────────────────────────────
        max_probes: int = 6,             # max exploration directions before giving up
        probe_warmup_batches: int = 10,  # batches at high LR per probe
        probe_anneal_batches: int = 30,  # batches to anneal back down per probe
        probe_eval_batches: int = 20,    # batches to evaluate if new basin is better

        exploration_lr_multiplier: float = 10.0,  # how much to boost LR for exploration
        min_exploration_multiplier: float = 2.0,   # minimum boost (decays over probes)
        perturbation_scale: float = 0.01,          # scale of random parameter perturbation

        # ── Acceptance criteria ─────────────────────────────────
        improvement_threshold: float = 0.02,  # require 2% improvement to accept new basin
        consistency_required: int = 5,         # consecutive improving eval batches needed

        # ── Base LR schedule ────────────────────────────────────
        base_lr: Optional[float] = None,
        min_lr: float = 1e-6,

        # ── Logging ─────────────────────────────────────────────
        verbose: bool = True,

        last_epoch: int = -1,
    ):
        self.model = model
        self.patience = patience
        self.plateau_threshold = plateau_threshold
        self.loss_window = loss_window

        self.max_probes = max_probes
        self.probe_warmup_batches = probe_warmup_batches
        self.probe_anneal_batches = probe_anneal_batches
        self.probe_eval_batches = probe_eval_batches

        self.exploration_lr_multiplier = exploration_lr_multiplier
        self.min_exploration_multiplier = min_exploration_multiplier
        self.perturbation_scale = perturbation_scale

        self.improvement_threshold = improvement_threshold
        self.consistency_required = consistency_required

        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.min_lr = min_lr
        self.verbose = verbose

        # ── Internal state ──────────────────────────────────────
        self._loss_history = deque(maxlen=loss_window * 2)
        self._state = "normal"  # normal | exploring_warmup | exploring_anneal | evaluating
        self._home_base = None  # saved (model_state, optimizer_state, loss)
        self._current_probe = 0
        self._probe_step = 0
        self._eval_losses = []
        self._plateau_counter = 0
        self._best_known_loss = float('inf')
        self._exploration_history = []  # track which probes succeeded/failed
        self._current_lr = self.base_lr
        self._total_steps = 0
        self._probes_since_last_improvement = 0

        # Direction tracking: we use different random seeds for each probe
        # to implicitly explore different directions in parameter space
        self._probe_rng_seeds = []

        super().__init__(optimizer, last_epoch)

    def _save_home_base(self, current_loss: float):
        """Save the current state as our 'home base' to return to."""
        self._home_base = {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'loss': current_loss,
            'lr': self._current_lr,
            'step': self._total_steps,
        }
        if self.verbose:
            print(f"[MinimaExplorer] 📍 Home base saved at loss={current_loss:.6f}")

    def _restore_home_base(self):
        """Roll back to the saved home base."""
        if self._home_base is None:
            return
        self.model.load_state_dict(self._home_base['model_state'])
        self.optimizer.load_state_dict(self._home_base['optimizer_state'])
        self._current_lr = self._home_base['lr']
        self._apply_lr(self._current_lr)
        if self.verbose:
            print(f"[MinimaExplorer] 🔙 Rolled back to home base "
                  f"(loss={self._home_base['loss']:.6f})")

    def _apply_lr(self, lr: float):
        """Set the learning rate on all parameter groups."""
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _detect_plateau(self) -> bool:
        """
        Detect if we're in a local minimum by checking if the loss
        has stopped improving meaningfully.

        Uses two windows: a recent window and an older window.
        If the recent window's mean is not significantly better than
        the older window's mean, we're on a plateau.
        """
        if len(self._loss_history) < self.loss_window * 2:
            return False

        recent = list(self._loss_history)[-self.loss_window:]
        older = list(self._loss_history)[-self.loss_window*2:-self.loss_window]

        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)

        # Also check variance — very low variance means we're stuck
        recent_var = sum((x - recent_mean)**2 for x in recent) / len(recent)

        # Relative improvement
        if older_mean > 0:
            relative_improvement = (older_mean - recent_mean) / older_mean
        else:
            relative_improvement = 0.0

        is_plateau = (
            relative_improvement < self.plateau_threshold
            and recent_var < (recent_mean * 0.01) ** 2  # variance < 1% of mean squared
        )

        return is_plateau

    def _perturb_parameters(self, probe_index: int):
        """
        Add a small random perturbation to model parameters.

        Each probe uses a different random seed, so it effectively
        explores a different direction in parameter space. This is
        the key mechanism for "looking around in all directions."

        The perturbation is scaled relative to each parameter's magnitude,
        so it respects the natural scale of different layers.
        """
        seed = hash(f"probe_{probe_index}_{self._total_steps}") % (2**32)
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._probe_rng_seeds.append(seed)

        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad:
                    # Scale perturbation relative to parameter magnitude
                    param_scale = param.data.abs().mean().item() + 1e-8
                    noise = torch.randn_like(param.data, generator=rng)
                    param.data += noise * self.perturbation_scale * param_scale

    def _reset_optimizer_momentum(self):
        """
        Reset the optimizer's momentum/adaptive state.

        This is crucial: by clearing the accumulated gradients and
        momentum, the optimizer will follow a fresh gradient direction
        from the perturbed position, effectively exploring a new
        trajectory through the loss landscape.
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state.get(p, {})
                # Reset Adam/AdamW state
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'exp_avg_sq' in state:
                    state['exp_avg_sq'].zero_()
                # Reset SGD momentum
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].zero_()

    def _get_exploration_lr(self) -> float:
        """
        Compute the exploration learning rate for the current probe.

        Earlier probes use larger multipliers (explore far).
        Later probes use smaller multipliers (explore nearby).
        This creates a coarse-to-fine search pattern.
        """
        # Decay the multiplier across probes
        decay = self._current_probe / max(self.max_probes - 1, 1)
        multiplier = (
            self.exploration_lr_multiplier * (1 - decay)
            + self.min_exploration_multiplier * decay
        )
        return self._current_lr * multiplier

    def _get_annealing_lr(self, step_in_anneal: int) -> float:
        """
        Cosine annealing from exploration LR back down to base LR.
        This is the "large → small" part of each probe cycle.
        """
        exploration_lr = self._get_exploration_lr()
        progress = step_in_anneal / max(self.probe_anneal_batches, 1)
        # Cosine decay from exploration_lr to current_lr
        return self._current_lr + 0.5 * (exploration_lr - self._current_lr) * (
            1 + math.cos(math.pi * progress)
        )

    def report_loss(self, loss: float):
        """
        Call this after each training batch with the current loss.
        This is the main interface — it drives the state machine.
        """
        self._loss_history.append(loss)
        self._total_steps += 1

        if loss < self._best_known_loss:
            self._best_known_loss = loss

        # ── State machine ───────────────────────────────────────
        if self._state == "normal":
            self._handle_normal(loss)
        elif self._state == "exploring_warmup":
            self._handle_exploring_warmup(loss)
        elif self._state == "exploring_anneal":
            self._handle_exploring_anneal(loss)
        elif self._state == "evaluating":
            self._handle_evaluating(loss)

    def _handle_normal(self, loss: float):
        """Normal training — watch for plateaus."""
        if self._detect_plateau():
            self._plateau_counter += 1
            if self._plateau_counter >= self.patience:
                self._begin_exploration(loss)
                self._plateau_counter = 0
        else:
            self._plateau_counter = max(0, self._plateau_counter - 1)

    def _begin_exploration(self, current_loss: float):
        """Transition from normal training to exploration mode."""
        if self.verbose:
            print(f"\n[MinimaExplorer] 🔍 Plateau detected at loss={current_loss:.6f}. "
                  f"Beginning exploration (up to {self.max_probes} probes)...")

        self._save_home_base(current_loss)
        self._current_probe = 0
        self._probes_since_last_improvement = 0
        self._start_probe()

    def _start_probe(self):
        """Start a new exploration probe."""
        if self._current_probe >= self.max_probes:
            # Exhausted all probes — return home and resume normal training
            self._restore_home_base()
            self._state = "normal"
            self._probes_since_last_improvement += 1

            if self.verbose:
                print(f"[MinimaExplorer] 🏠 All {self.max_probes} probes exhausted. "
                      f"Returning to home base and continuing normal training.")

            # Optionally reduce base LR slightly after failed exploration
            # (the minimum is getting harder to escape — try finer steps)
            if self._probes_since_last_improvement >= 2:
                self._current_lr = max(self.min_lr, self._current_lr * 0.8)
                self._apply_lr(self._current_lr)
                if self.verbose:
                    print(f"[MinimaExplorer] 📉 Reduced base LR to {self._current_lr:.2e} "
                          f"after {self._probes_since_last_improvement} failed exploration rounds")
            return

        if self.verbose:
            print(f"[MinimaExplorer] 🚀 Probe {self._current_probe + 1}/{self.max_probes} — "
                  f"LR boost: {self._get_exploration_lr():.2e}")

        # Restore home base before each probe (clean starting point)
        if self._current_probe > 0:
            self._restore_home_base()

        # Perturb parameters to explore a different direction
        self._perturb_parameters(self._current_probe)

        # Reset optimizer momentum for fresh gradient following
        self._reset_optimizer_momentum()

        # Set high learning rate for warmup phase
        exploration_lr = self._get_exploration_lr()
        self._apply_lr(exploration_lr)

        self._state = "exploring_warmup"
        self._probe_step = 0

    def _handle_exploring_warmup(self, loss: float):
        """High LR phase — escape the current basin."""
        self._probe_step += 1

        if self._probe_step >= self.probe_warmup_batches:
            # Transition to annealing phase
            self._state = "exploring_anneal"
            self._probe_step = 0
            if self.verbose:
                print(f"[MinimaExplorer] 📐 Probe {self._current_probe + 1}: "
                      f"warmup done, annealing... (loss={loss:.6f})")

    def _handle_exploring_anneal(self, loss: float):
        """Cosine annealing phase — settle into a new basin."""
        self._probe_step += 1

        # Apply cosine annealing
        lr = self._get_annealing_lr(self._probe_step)
        self._apply_lr(lr)

        if self._probe_step >= self.probe_anneal_batches:
            # Transition to evaluation phase
            self._state = "evaluating"
            self._probe_step = 0
            self._eval_losses = []
            self._apply_lr(self._current_lr)  # back to base LR for fair eval
            if self.verbose:
                print(f"[MinimaExplorer] 📊 Probe {self._current_probe + 1}: "
                      f"annealing done, evaluating... (loss={loss:.6f})")

    def _handle_evaluating(self, loss: float):
        """Evaluate whether the new basin is better than home base."""
        self._eval_losses.append(loss)
        self._probe_step += 1

        if self._probe_step >= self.probe_eval_batches:
            self._judge_probe()

    def _judge_probe(self):
        """
        Decide whether to accept the new basin or try another probe.

        Acceptance criteria:
        1. Mean eval loss must be significantly better than home base
        2. The loss trajectory during eval must be consistently improving
           (not just a lucky dip)
        """
        if not self._eval_losses or self._home_base is None:
            self._reject_probe("no eval data")
            return

        home_loss = self._home_base['loss']
        eval_mean = sum(self._eval_losses) / len(self._eval_losses)
        eval_min = min(self._eval_losses)

        # Check 1: Is the mean significantly better?
        if home_loss > 0:
            relative_improvement = (home_loss - eval_mean) / home_loss
        else:
            relative_improvement = 0.0

        # Check 2: Is the trajectory consistently improving?
        # Count how many consecutive losses at the end are below home_loss
        consecutive_better = 0
        for loss in reversed(self._eval_losses):
            if loss < home_loss:
                consecutive_better += 1
            else:
                break

        # Check 3: Is the trend downward? (linear regression slope)
        if len(self._eval_losses) >= 3:
            xs = list(range(len(self._eval_losses)))
            x_mean = sum(xs) / len(xs)
            y_mean = eval_mean
            numerator = sum((x - x_mean) * (y - y_mean)
                          for x, y in zip(xs, self._eval_losses))
            denominator = sum((x - x_mean) ** 2 for x in xs)
            slope = numerator / (denominator + 1e-8)
            is_trending_down = slope < 0
        else:
            is_trending_down = False

        # ── Decision ────────────────────────────────────────────
        accept = (
            relative_improvement > self.improvement_threshold
            and consecutive_better >= self.consistency_required
            and is_trending_down
        )

        if accept:
            self._accept_probe(eval_mean, relative_improvement)
        else:
            reason = (
                f"improvement={relative_improvement:.4f} "
                f"(need>{self.improvement_threshold}), "
                f"consecutive_better={consecutive_better} "
                f"(need>={self.consistency_required}), "
                f"trending_down={is_trending_down}"
            )
            self._reject_probe(reason)

    def _accept_probe(self, new_loss: float, improvement: float):
        """Accept the new basin — update home base and resume normal training."""
        if self.verbose:
            print(f"[MinimaExplorer] ✅ Probe {self._current_probe + 1} ACCEPTED! "
                  f"New loss={new_loss:.6f} "
                  f"(improvement={improvement:.2%} over home base)")

        self._exploration_history.append({
            'probe': self._current_probe,
            'accepted': True,
            'old_loss': self._home_base['loss'],
            'new_loss': new_loss,
            'improvement': improvement,
            'step': self._total_steps,
        })

        # The current model state IS the new home base
        self._home_base = None
        self._state = "normal"
        self._probes_since_last_improvement = 0
        self._best_known_loss = min(self._best_known_loss, new_loss)

    def _reject_probe(self, reason: str):
        """Reject the current probe and try the next direction."""
        if self.verbose:
            print(f"[MinimaExplorer] ❌ Probe {self._current_probe + 1} rejected: {reason}")

        self._exploration_history.append({
            'probe': self._current_probe,
            'accepted': False,
            'reason': reason,
            'step': self._total_steps,
        })

        self._current_probe += 1
        self._start_probe()  # This handles the "all probes exhausted" case

    def get_lr(self) -> List[float]:
        """Required by _LRScheduler — returns current LR for each param group."""
        return [self._current_lr] * len(self.optimizer.param_groups)

    def get_state(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            'state': self._state,
            'current_lr': self._current_lr,
            'best_known_loss': self._best_known_loss,
            'current_probe': self._current_probe,
            'probe_step': self._probe_step,
            'plateau_counter': self._plateau_counter,
            'total_steps': self._total_steps,
            'exploration_history': self._exploration_history,
            'probes_since_last_improvement': self._probes_since_last_improvement,
            # Note: home_base contains model/optimizer state dicts which are large
            # For checkpointing, you might want to save these separately
        }

    @property
    def is_exploring(self) -> bool:
        return self._state != "normal"

    @property
    def exploration_summary(self) -> str:
        n_accepted = sum(1 for h in self._exploration_history if h.get('accepted'))
        n_total = len(self._exploration_history)
        return (f"Explorations: {n_total} probes, {n_accepted} accepted, "
                f"best_loss={self._best_known_loss:.6f}")

SCHEDULERS = {
    "cosine": lambda opt, ep: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep),
    "step": lambda opt, ep: torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, ep // 3), gamma=0.5),
    "plateau": lambda opt, ep: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5),
    "none": lambda opt, ep: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 1.0),
    "warmup_cosine": None,
    "minima_explorer": None
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
    g.add_argument("--batches-per-epoch", type=int, default=300,
                   help="Training batches per epoch")
    g.add_argument("--val-batches", type=int, default=200,
                   help="Validation batches per epoch")
    g.add_argument("--max-params", type=int, default=3,
                   help="Max function parameters")
    g.add_argument("--max-ops", type=int, default=2,
                   help="Max operations in random DAG")
    g.add_argument("--allowed-ops", type=str, default="add,sub",
                   help="Comma-separated LLVM ops")
    g.add_argument("--param-min", type=int, default=-50,
                   help="Min random parameter value")
    g.add_argument("--param-max", type=int, default=50,
                   help="Max random parameter value")
    g.add_argument("--task", type=str, default="infix",
                   choices=["infix", "turnstile"],
                   help="Task type: 'infix' = current math expressions, "
                        "'turnstile' = !-counting equality evaluation (0/1)")
    g.add_argument("--max-turnstiles", type=int, default=10,
                   help="Max ! symbols on each side (turnstile task only)")

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
    g.add_argument("--ffn", type=int, default=0,
                   help="MLP/FFN hidden dimension (0 = auto, defaults to 4*d_model)")

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
    g.add_argument("--structure-loss-alpha", type=float, default=0.0,
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
    g.add_argument("--continue", type=str, default=None, dest="continue_run",
                   help="Path to a run directory to continue from (e.g. runs/0). "
                   "Loads the last checkpoint and resumes with identical settings.")
    g.add_argument("--wait-pid", type=int, default=None,
                   help="Wait for this PID to exit before starting training "
                   "(useful for queueing CUDA jobs)")
    g.add_argument("--sync", type=str, default=None, metavar="USER@SERVER:/PATH",
                   help="After each epoch, run watch_and_sync.sh in the background "
                   "to sync the run directory to a remote server. "
                   "Example: --sync root@myserver.de:/var/www/grok_test/  "
                   "Only one sync process runs at a time.")

    g = p.add_argument_group("Topology")
    g.add_argument("--topo", action="store_true", default=False,
                   help="Enable live topological barcode visualization")
    g.add_argument("--kelp-every", type=int, default=25,
                   help="Update topological kelp every N batches")
    g.add_argument("--topo-every", type=int, default=50,
                   help="Update topological barcodes every N batches")
    g.add_argument("--topo-max-points", type=int, default=200,
                   help="Max points to subsample for persistence computation")

    g = p.add_argument_group("Replay Buffer")
    g.add_argument("--replay-save-rate", type=float, default=0.7,
                   help="Probability of saving each generated sample to the replay buffer. "
                   "0 = disabled, 0.10 = save ~10%% of samples.")
    g.add_argument("--replay-sprinkle-rate", type=float, default=0.7,
                   help="Fraction of each batch to fill from the replay buffer "
                   "(only active when replay-save-rate > 0)")
    g.add_argument("--replay-max-size", type=int, default=200_000,
                   help="Maximum number of samples to retain in the replay buffer")

    return p.parse_args()

class ReplayBuffer:
    """
    Stores a fraction of generated programs and mixes them back into
    future batches.
    """

    def __init__(self, save_rate: float = 0.10, sprinkle_rate: float = 0.10,
                 max_size: int = 5000, decay_rate: float = 0.001):
        self.save_rate = save_rate
        self.sprinkle_rate = sprinkle_rate
        self.max_size = max_size
        self.decay_rate = decay_rate  # fraction of buffer to randomly delete
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._total_saved = 0
        self._total_sprinkled = 0
        self._total_decayed = 0

    def maybe_save(self, samples: List[Tuple[List[int], int]]):
        with self._lock:
            for sample in samples:
                if random.random() < self.save_rate:
                    self._buffer.append(sample)
                    self._total_saved += 1
            # After saving new samples, randomly purge 0.1% of the buffer
            self._random_decay()

    def _random_decay(self):
        """
        Randomly delete `decay_rate` (default 0.1%) of the buffer entries.
        This forces those slots to be re-filled by fresh randomly generated
        samples, preventing the buffer from going stale.

        Must be called while self._lock is held.
        """
        if not self._buffer or self.decay_rate <= 0:
            return

        n_to_delete = max(1, int(len(self._buffer) * self.decay_rate))

        # Only bother if the buffer is large enough that deletion is meaningful
        if n_to_delete < 1 or len(self._buffer) <= n_to_delete:
            return

        # Pick random indices to remove (without replacement)
        indices_to_remove = set(random.sample(range(len(self._buffer)), n_to_delete))

        # Rebuild the deque without the selected indices
        self._buffer = deque(
            (item for i, item in enumerate(self._buffer) if i not in indices_to_remove),
            maxlen=self.max_size,
        )
        self._total_decayed += n_to_delete

    def sprinkle(self, batch_size: int) -> List[Tuple[List[int], int]]:
        n_sprinkle = int(batch_size * self.sprinkle_rate)
        if n_sprinkle == 0 or not self._buffer:
            return []

        with self._lock:
            n_sprinkle = min(n_sprinkle, len(self._buffer))
            buffer_list = list(self._buffer)
            chosen = random.sample(buffer_list, n_sprinkle)
            self._total_sprinkled += n_sprinkle
            return chosen

    def get_state(self) -> dict:
        with self._lock:
            return {
                "buffer": list(self._buffer),
                "total_saved": self._total_saved,
                "total_sprinkled": self._total_sprinkled,
                "total_decayed": self._total_decayed,
            }

    def restore_state(self, state: dict):
        if not state:
            return
        with self._lock:
            self._buffer = deque(state.get("buffer", []), maxlen=self.max_size)
            self._total_saved = state.get("total_saved", 0)
            self._total_sprinkled = state.get("total_sprinkled", 0)
            self._total_decayed = state.get("total_decayed", 0)
        console.print(
            f"  [green]✓ Replay buffer restored: {len(self._buffer)} samples[/]"
        )

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "buffer_size": len(self._buffer),
                "total_saved": self._total_saved,
                "total_sprinkled": self._total_sprinkled,
                "total_decayed": self._total_decayed,
            }

args = None

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
    while _pid_alive(pid):
        time.sleep(poll_interval)

    console.print(f"[bold green]✅ PID {pid} is done. Proceeding to training.[/]")


def _pid_alive(pid: int) -> bool:
    """Return True if pid is still in the process table."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it — still alive
        return True
    except OSError:
        return False
    return True

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

    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def __init__(self, hf_tokenizer: HFTokenizer = None):
        self._tok = hf_tokenizer
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None
        if hf_tokenizer is not None:
            self._cache_special_ids()

    def _cache_special_ids(self):
        self._pad_id = self._tok.token_to_id("<pad>")
        self._bos_id = self._tok.token_to_id("<bos>")
        self._eos_id = self._tok.token_to_id("<eos>")

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
        ) -> Optional[Tuple[List[int], int]]:
    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

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

    result_str = str(result)
    # ir_code already ends with "= ", so just append the result
    text = f"{ir_code}{result_str}"

    # ── FIX: Compute prompt_len from the JOINT encoding ─────────────
    # Encode the full text as one unit, then find where the answer
    # starts by encoding the prompt alone and searching for the
    # divergence point. This avoids BPE boundary mismatches.
    full_ids = tokenizer.encode(text)
    prompt_only_ids = tokenizer.encode(ir_code)

    # Find the longest common prefix between prompt_only_ids and full_ids.
    # The answer starts right after this prefix in full_ids.
    common_len = 0
    for a, b in zip(prompt_only_ids, full_ids):
        if a == b:
            common_len += 1
        else:
            break
    else:
        # All of prompt_only_ids matched — answer starts right after
        common_len = len(prompt_only_ids)

    # prompt_len includes the <bos> token
    prompt_len = 1 + common_len  # +1 for <bos>

    # ── Validate: the answer portion must be non-empty ──────────────
    # full sequence = [bos] + full_ids + [eos]
    # answer = full_ids[common_len:]
    answer_ids = full_ids[common_len:]
    if not answer_ids:
        # BPE merged the entire answer into the last prompt token.
        # Fall back: decode and re-search for the result string.
        # Find the result_str in the decoded full text token by token.
        accumulated = ""
        prompt_len_fallback = 1  # start after <bos>
        for i, tid in enumerate(full_ids):
            accumulated = tokenizer.decode(full_ids[:i + 1])
            # Check if we've reached the end of the prompt portion
            if accumulated.rstrip().endswith("= ") or accumulated.rstrip().endswith("="):
                # Check if the remaining tokens encode the answer
                remaining = tokenizer.decode(full_ids[i + 1:]).strip()
                if remaining == result_str or remaining.startswith(result_str):
                    prompt_len_fallback = 1 + (i + 1)
                    break
        prompt_len = prompt_len_fallback

        # Final safety: ensure answer portion is non-empty
        total_len = 1 + len(full_ids) + 1  # bos + tokens + eos
        if prompt_len >= total_len - 1:
            # Cannot recover — skip this sample
            return None

    ids = (
            [tokenizer.bos_token_id]
            + full_ids
            + [tokenizer.eos_token_id]
            )

    return ids, prompt_len

def generate_batch(
    tokenizer: BPETokenizer,
    batch_size: int,
    max_params: int = 3,
    max_ops: int = 4,
    allowed_ops: Optional[List[str]] = None,
    param_range: Tuple[int, int] = (-20, 20),
    max_seq_len: int = 2048,
    task: str = "infix",              # ← NEW
    max_turnstiles: int = 10,         # ← NEW
) -> List[List[int]]:
    samples = []
    attempts = 0
    max_attempts = batch_size * 5

    while len(samples) < batch_size and attempts < max_attempts:
        attempts += 1

        if task == "turnstile":
            s = generate_single_sample_turnstile(
                tokenizer, max_turnstiles=max_turnstiles,
                max_seq_len=max_seq_len,
            )
        else:  # "infix"
            s = generate_single_sample(
                tokenizer, max_params, max_ops, allowed_ops,
                param_range, max_seq_len,
            )

        if s is not None:
            samples.append(s)

    if len(samples) < batch_size and len(samples) > 0:
        console.print(
            f"  [dim yellow]⚠ Only generated {len(samples)}/{batch_size} samples "
            f"after {max_attempts} attempts[/]"
        )

    return samples

def collate_batch(batch, pad_id=0, tokenizer=None):
    """
    Classification collation for the turnstile task.

    Each sample is (token_ids, prompt_len) where:
      - token_ids = [bos] + encoded(prompt + answer) + [eos]
      - prompt_len = number of tokens up to and including the prompt

    We extract:
      - input_ids: just the prompt portion (padded), NO answer tokens
      - labels: integer 0 or 1 parsed from the answer portion

    Returns:
        input_ids:  (B, T)  LongTensor — prompt tokens only, padded
        labels:     (B,)    LongTensor — 0 or 1
    """
    eos_id = tokenizer.eos_token_id if tokenizer else None

    input_seqs = []
    labels = []

    for token_ids, prompt_len in batch:
        # ── Extract prompt tokens (including <bos>) ─────────────────
        prompt_ids = token_ids[:prompt_len]
        input_seqs.append(prompt_ids)

        # ── Extract answer from the token sequence ──────────────────
        # Answer tokens are between prompt_len and <eos>
        answer_end = len(token_ids)
        for i in range(prompt_len, len(token_ids)):
            if token_ids[i] == eos_id:
                answer_end = i
                break

        answer_ids = token_ids[prompt_len:answer_end]

        # Decode answer to get the label
        label = 0  # default
        if tokenizer and answer_ids:
            answer_str = tokenizer.decode(answer_ids).strip()
            # Clean any special token artifacts
            for special in ["<eos>", "<pad>", "<bos>", "<sep>"]:
                answer_str = answer_str.replace(special, "")
            answer_str = ''.join(
                c for c in answer_str if c.isascii() and c.isprintable()
            ).strip()
            try:
                label = int(answer_str)
                # Clamp to valid range for safety
                if label not in (0, 1):
                    label = 0
            except (ValueError, TypeError):
                label = 0

        labels.append(label)

    # ── Pad input sequences ─────────────────────────────────────────
    max_len = max(len(s) for s in input_seqs)
    padded_inputs = []
    for s in input_seqs:
        padded = s + [pad_id] * (max_len - len(s))
        padded_inputs.append(padded)

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

def build_tokenizer_from_samples(n_programs=1000, allowed_ops=None,
                                 max_params=4, max_ops=6,
                                 param_range=(-50, 50),
                                 bpe_vocab_size=512,
                                 task_type="infix") -> BPETokenizer:
    """
    Generate n_programs random functions to build a tokenizer.
    For turnstile task: build a character-level tokenizer with only the
    characters that actually appear in the data (no BPE, no byte-level).
    For infix task: standard BPE with byte-level pre-tokenization.
    """
    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    # ── Step 1: Generate corpus ─────────────────────────────────────────
    corpus: List[str] = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Generating corpus for tokenizer..."),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("• [cyan]{task.fields[status]}[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
            ) as progress:
        ptask = progress.add_task(
                "gen_corpus", total=n_programs, status="starting..."
                )
        success = 0
        fail = 0

        for i in range(n_programs):
            try:
                if task_type == "turnstile":
                    from turnstile_gen import generate_turnstile_function
                    prompt, result = generate_turnstile_function(max_turnstiles=10)
                    text = f"{prompt}{result}"
                    corpus.append(text)
                    success += 1
                else:
                    num_p = random.randint(2, max(2, max_params))
                    num_o = random.randint(1, max(1, max_ops))
                    params = [random.randint(*param_range) for _ in range(num_p)]
                    ir_code, result = generate_random_function(
                            num_params=num_p, params=params,
                            allowed_ops=allowed_ops, num_operations=num_o,
                            func_name="f",
                            )
                    text = f"{ir_code}{result}"
                    corpus.append(text)
                    success += 1
            except Exception:
                fail += 1

            progress.update(
                    ptask, advance=1,
                    status=f"{success} ok, {fail} failed"
                    )

    console.print(
            f"  [green]✓[/] Corpus ready: [bold]{len(corpus)}[/] programs "
            f"([dim]{fail} generation failures skipped[/])"
            )

    # ── Step 2: Build tokenizer ─────────────────────────────────────────

    special_tokens = ["<pad>", "<bos>", "<eos>"]

    if task_type == "turnstile":
        # ── Character-level tokenizer: only the chars that appear ────────
        charset = set()
        for text in corpus:
            charset.update(text)
        alphabet = sorted(charset)

        console.print(
            f"  [cyan]Turnstile task: building character-level tokenizer "
            f"(alphabet={len(alphabet)} unique chars, no BPE merges)...[/]"
        )

        # Build vocab: special tokens first, then each character
        vocab = {}
        for i, tok in enumerate(special_tokens):
            vocab[tok] = i
        for ch in alphabet:
            if ch not in vocab:
                vocab[ch] = len(vocab)

        # Construct a BPE model with an explicit vocab and NO merges
        hf_tokenizer = HFTokenizer(models.BPE(vocab=vocab, merges=[]))

        # Split into individual characters using a pattern compatible
        # with Oniguruma (the regex engine used by the tokenizers library).
        # [\s\S] matches any character including newlines (Oniguruma-safe
        # alternative to (?s).)
        from tokenizers import Regex, AddedToken
        hf_tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=Regex(r"[\s\S]"),
            behavior="isolated",
        )

        # Decoder: just fuse tokens back together
        hf_tokenizer.decoder = decoders.Fuse()

        # Register special tokens
        hf_tokenizer.add_special_tokens([
            AddedToken(tok, special=True) for tok in special_tokens
        ])

    else:
        # ── Standard BPE for infix task ─────────────────────────────────
        hf_tokenizer = HFTokenizer(models.BPE())
        hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        hf_tokenizer.decoder = decoders.ByteLevel()

        console.print(
                f"  [cyan]Training BPE tokenizer "
                f"(vocab_size={bpe_vocab_size}, corpus={len(corpus)} programs)...[/]"
                )

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
            ptask2 = progress.add_task("bpe_train", total=None)
            hf_tokenizer.train_from_iterator(corpus, trainer=trainer_obj)
            progress.update(ptask2, completed=True)

    tokenizer = BPETokenizer(hf_tokenizer=hf_tokenizer)

    console.print(
            f"  [green]✓[/] Tokenizer ready — "
            f"vocab_size=[bold]{tokenizer.vocab_size}[/]"
            )

    # Show a sample encoding
    if corpus:
        sample = corpus[0][:80]
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded)
        console.print(
                f"  [dim]Sample: \"{sample}...\"[/]\n"
                f"  [dim]→ {len(encoded)} tokens (vs {len(sample)} chars)[/]\n"
                f"  [dim]→ IDs: {encoded[:20]}{'...' if len(encoded) > 20 else ''}[/]\n"
                f"  [dim]→ Decoded: \"{decoded[:80]}\"[/]"
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
            ffn: int = 0,
            **kwargs,
            ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.ffn = ffn if ffn > 0 else 4 * d_model
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
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, ffn: int = 8):
        super().__init__()
        if not d_model % n_heads == 0:
            example_d_model = n_heads * ((d_model // n_heads)) + 1

            console.print(
                f"[red]" +
                f"Invalid configuration: d_model ({d_model}) must be divisible by n_heads ({n_heads}). " +
                f"Try setting d_model to a multiple of n_heads, such as {example_d_model}." +
                f"Auto setting ." +
                f"[/]"
            )

            d_model = example_d_model

        self.n_heads = n_heads
        self.ffn = ffn
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # RoPE instead of learned/absolute positional embeddings
        self.rotary_emb = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
        )

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Apply RoPE to queries and keys
            cos, sin = self.rotary_emb(T)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

            # Standard scaled dot-product attention
            scale = math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) / scale
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            out = (attn @ v).transpose(1, 2).reshape(B, T, C)
            return self.proj_drop(self.proj(out))
        except torch.cuda.OutOfMemoryError:
            console.print(
                f"[bold red]❌ Memory error. This can be caused by having a too large "
                f"batch size or too little parameters.[/]"
            )
            sys.exit(1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1, ffn: int = 8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout, ffn)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
                nn.Linear(d_model, ffn),
                nn.GELU(),
                nn.Linear(ffn, d_model),
                nn.Dropout(dropout),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyClassifierGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        # Token embedding only — NO positional embedding (RoPE is in attention)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads,
                              config.max_seq_len, dropout=0.0, ffn=config.ffn)
             for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.cls_head = nn.Linear(config.d_model, 2)
        self.apply(self._init_weights)

    def forward(self, input_ids, labels=None, output_hidden_states=False,
                value_positions=None, **kwargs):
        B, T = input_ids.shape

        # Only token embeddings — position is encoded via RoPE in each attention layer
        x = self.tok_emb(input_ids)

        hidden_states = [x] if output_hidden_states else None

        for blk in self.blocks:
            x = blk(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)

        # Pool: last token's hidden state
        pooled = x[:, -1, :]  # (B, d_model)
        logits = self.cls_head(pooled)  # (B, 2)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if output_hidden_states:
            return _ModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=tuple(hidden_states),
                last_hidden_state=x,
                value_preds=None,
            )

        return loss, logits

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.config.save_pretrained(path)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "TinyClassifierGPT":
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
        model: 'TinyClassifierGPT',
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
        model: 'TinyClassifierGPT',
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

def estimate_params(vocab_size: int, d_model: int, n_layers: int, max_seq_len: int, ffn: int = 0) -> int:
    actual_ffn = ffn if ffn > 0 else 4 * d_model
    # Token embedding only (no positional embedding with RoPE)
    emb = vocab_size * d_model
    # Transformer blocks:
    #   - QKV + proj: 4 * d_model^2 + 4 * d_model (biases)
    #   - MLP: d_model * actual_ffn + actual_ffn + actual_ffn * d_model + d_model
    #   - LayerNorms: 2 * (2 * d_model)
    attn_params = 4 * d_model * d_model + 4 * d_model  # qkv + proj
    mlp_params = d_model * actual_ffn + actual_ffn + actual_ffn * d_model + d_model
    ln_params = 4 * d_model  # 2 layer norms per block
    blocks = n_layers * (attn_params + mlp_params + ln_params)
    # Final layer norm
    ln_f = 2 * d_model
    # Classification head
    cls_head = d_model * 2 + 2
    return emb + blocks + ln_f + cls_head

def find_model_config(
        vocab_size: int,
        target_params: int = 1_000,
        max_seq_len: int = 2048,
        ffn: int = 0,
        ) -> dict:
    best = None
    best_diff = float("inf")
    for d_model in range(8, 512, 2):
        for n_heads in (1, 2, 4, 8):
            if d_model % n_heads != 0:
                continue
            for n_layers in range(1, 24):
                n = estimate_params(vocab_size, d_model, n_layers, max_seq_len, ffn)
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

# ════════════════════════════════════════════════════════════════════════════
# 7.  LIVE PLOTTER
# ════════════════════════════════════════════════════════════════════════════

def _prediction_error_score_turnstile(exp_val: int, pred_val: int) -> float:
    """
    Turnstile binary task scoring.

    Score range [0, 0.9] for valid binary outputs:
        0.0  — correct (pred == exp, both in {0, 1})
        0.9  — wrong binary (pred in {0, 1} but pred != exp)

    For non-binary integers: 1.0 (totally wrong)
    Unparseable strings are handled by the caller (also 1.0).
    """
    if pred_val == exp_val and pred_val in (0, 1):
        return 0.0

    if pred_val in (0, 1):
        return 0.9

    # Any other integer (2, -5, 137, etc.) is garbage
    return 1.0

def _prediction_error_score(exp_val: int, pred_val: int) -> float:
    """
    Score in [0, 0.9] where 0 = perfect, higher = worse.
    
    Uses symmetric relative error so that:
      - 0 vs 0   → 0.0  (perfect)
      - -0 vs 0  → 0.0  (numerically identical)
      - -10 vs 10 → small score (magnitudes match, only sign differs)
      - 5 vs 500  → large score (magnitudes very different)
    """
    # Treat as numerically identical (handles -0 vs 0 too)
    if exp_val == pred_val:
        return 0.0
    
    abs_exp = abs(exp_val)
    abs_pred = abs(pred_val)
    
    # Magnitude closeness: symmetric relative error on absolute values
    # This makes |10| vs |10| = 0 even if signs differ
    mag_diff = abs(abs_exp - abs_pred)
    mag_scale = max(abs_exp, abs_pred, 1)  # avoid div-by-zero
    magnitude_error = mag_diff / mag_scale  # in [0, 1+]
    
    # Sign penalty: 0 if same sign (or either is zero), small penalty otherwise
    exp_sign = (exp_val > 0) - (exp_val < 0)
    pred_sign = (pred_val > 0) - (pred_val < 0)
    if exp_sign == 0 or pred_sign == 0:
        sign_penalty = 0.0
    elif exp_sign != pred_sign:
        sign_penalty = 0.15  # fixed small penalty for wrong sign
    else:
        sign_penalty = 0.0
    
    # Combine: magnitude dominates, sign is a secondary signal
    # Saturate magnitude_error with 1 - exp(-k*x) curve
    mag_score = 0.75 * (1.0 - math.exp(-2.0 * magnitude_error))
    
    score = mag_score + sign_penalty
    return min(0.9, score)

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


from liveplotter import LivePlotter
import liveplotter as _lp_module
_lp_module._suppress_c_stderr = _suppress_c_stderr
_lp_module._restore_c_stderr = _restore_c_stderr
_lp_module._prediction_error_score = _prediction_error_score
_lp_module._is_int_str = _is_int_str
_lp_module.get_gpu_info = get_gpu_info
_lp_module.compute_persistence_landscapes = compute_persistence_landscapes
_lp_module.compute_cross_layer_wasserstein = compute_cross_layer_wasserstein
_lp_module._HAS_RIPSER = _HAS_RIPSER
_lp_module._HAS_GUDHI = _HAS_GUDHI

@torch.no_grad()
def get_batch_predictions(model, tokenizer, batch, device, max_gen_len=20, task="infix"):
    """
    Classification prediction: run the prompt through the model,
    argmax the 2-class logits, compare to expected label.
    
    Also handles autoregressive models gracefully by detecting
    whether the model outputs 2D (classifier) or 3D (LM) logits.
    """
    model.eval()
    predictions = []

    eos_id = tokenizer.eos_token_id
    is_classifier = hasattr(model, 'cls_head')

    for token_ids, prompt_len in batch[:8]:
        # ── Extract expected answer ─────────────────────────────────
        answer_end = len(token_ids)
        for i in range(prompt_len, len(token_ids)):
            if token_ids[i] == eos_id:
                answer_end = i
                break

        expected_ids = token_ids[prompt_len:answer_end]
        if expected_ids:
            expected_answer = tokenizer.decode(expected_ids).strip()
        else:
            expected_answer = ""

        # Clean
        for special in ["<eos>", "<pad>", "<bos>", "<sep>"]:
            expected_answer = expected_answer.replace(special, "")
        expected_answer = ''.join(
            c for c in expected_answer if c.isascii() and c.isprintable()
        ).strip()

        if not expected_answer:
            continue

        prompt_ids = token_ids[:prompt_len]
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        if is_classifier or task == "turnstile":
            # ── Classification: feed prompt, argmax logits ──────────
            output = model(inp)
            if isinstance(output, tuple):
                _, logits = output
            else:
                logits = output.logits
            predicted_class = logits.argmax(dim=-1).item()
            predicted_answer = str(predicted_class)
        else:
            # ── Autoregressive decode ───────────────────────────────
            generated_ids = []
            input_tensor = inp
            for _ in range(max_gen_len):
                if input_tensor.shape[1] >= model.max_seq_len:
                    break
                output = model(input_ids=input_tensor, output_hidden_states=False)
                if isinstance(output, tuple):
                    _, step_logits = output
                else:
                    step_logits = output.logits

                # Safety: if 2D logits, treat as classifier
                if step_logits.dim() == 2:
                    next_id = step_logits.argmax(dim=-1).item()
                    if next_id != eos_id and next_id != tokenizer.pad_token_id:
                        generated_ids.append(next_id)
                    break

                next_logits = step_logits[:, -1, :]
                next_id = next_logits.argmax(dim=-1).item()

                if next_id == eos_id or next_id == tokenizer.pad_token_id:
                    break

                generated_ids.append(next_id)
                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_id]], device=device)],
                    dim=1,
                )

            if generated_ids:
                predicted_answer = tokenizer.decode(generated_ids).strip()
                for special in ["<eos>", "<pad>", "<bos>"]:
                    predicted_answer = predicted_answer.replace(special, "")
                predicted_answer = predicted_answer.strip()
            else:
                predicted_answer = ""

        if not predicted_answer:
            predicted_answer = "(empty)"

        try:
            is_correct = int(predicted_answer) == int(expected_answer)
        except (ValueError, TypeError):
            is_correct = predicted_answer == expected_answer

        predictions.append((expected_answer, predicted_answer, is_correct))

    model.train()
    return predictions

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
    replay_buffer: Optional['ReplayBuffer'] = None,
    task: str = "infix",
    max_turnstiles: int = 10,
) -> Optional[Tuple]:
    sprinkled = []
    if replay_buffer is not None:
        sprinkled = replay_buffer.sprinkle(batch_size)

    n_fresh = batch_size - len(sprinkled)

    batch = generate_batch(
        tokenizer, n_fresh, max_params, max_ops,
        allowed_ops, param_range, max_seq_len,
        task=task,
        max_turnstiles=max_turnstiles,
    )

    if len(batch) < 2 and not sprinkled:
        return None

    # Save fresh samples to the buffer BEFORE mixing
    if replay_buffer is not None and batch:
        replay_buffer.maybe_save(batch)

    # Combine fresh + sprinkled
    combined = batch + sprinkled
    random.shuffle(combined)

    if len(combined) < 2:
        return None

    inp, labels = collate_batch(
        combined,
        pad_id=tokenizer.SPECIAL["<pad>"],
        tokenizer=tokenizer,
    )

    return (
        combined,
        inp.to(device),
        labels.to(device),
    )

def _compute_batch_loss(
    model: TinyClassifierGPT,
    inp: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: BPETokenizer,
    device: str,
    value_loss_alpha: float = 0.0,
    structure_loss_alpha: float = 0.0,
    length_loss_alpha: float = 0.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Classification loss for the turnstile task.

    Returns:
        (loss_tensor, loss_scalar, accuracy)
    """
    loss, logits = model(inp, labels=labels)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()

    return loss, loss.item(), accuracy

def _save_checkpoint(
        path: str,
        filename: str,
        epoch: int,
        model: TinyClassifierGPT,
        optimizer,
        scheduler,
        train_loss: float,
        val_loss: float,
        best_val_loss: float,
        train_losses_hist: List[float] = None,
        val_losses_hist: List[float] = None,
        total_samples: int = 0,
        rng_state: dict = None,
        plotter_state: dict = None,
        replay_buffer_state: dict = None,
        args_dict: dict = None,
        ema_train: float = None,
        global_batch: int = 0,
        ) -> str:
    """Save a .pt checkpoint to path/filename using atomic write.
    Returns the full path."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    tmp_path = full_path + ".tmp"

    # Capture RNG states if not provided
    if rng_state is None:
        rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    # Write to a temporary file first
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "train_losses_hist": train_losses_hist or [],
        "val_losses_hist": val_losses_hist or [],
        "total_samples": total_samples,
        "rng_state": rng_state,
        "plotter_state": plotter_state or {},
        "replay_buffer_state": replay_buffer_state,
        "args_dict": args_dict or {},
        "ema_train": ema_train,
        "global_batch": global_batch,
        }, tmp_path)

    # Atomic rename: either the old file remains or the new one replaces it
    # This prevents corruption from Ctrl+C during write
    os.replace(tmp_path, full_path)

    return full_path

def _find_latest_checkpoint(run_path: str) -> Optional[str]:
    """Find the most recent VALID model_epoch_*.pt checkpoint in a run directory."""
    import glob
    pattern = os.path.join(run_path, "model_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None

    def _epoch_num(p):
        base = os.path.basename(p)
        try:
            return int(base.replace("model_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1

    checkpoints.sort(key=_epoch_num, reverse=True)

    for ckpt in checkpoints:
        try:
            # Light validation: just check it's a valid zip/tar archive
            # by loading only the keys, not the full tensors
            import zipfile
            if zipfile.is_zipfile(ckpt):
                with zipfile.ZipFile(ckpt, 'r') as zf:
                    names = zf.namelist()
                    if not names:
                        raise ValueError("Empty checkpoint archive")
            else:
                # Fallback: try loading with map_location=meta to avoid RAM usage
                torch.load(ckpt, map_location="meta", weights_only=False)
            return ckpt
        except Exception as e:
            epoch_n = _epoch_num(ckpt)
            console.print(
                    f"  [yellow]⚠ Checkpoint model_epoch_{epoch_n}.pt is corrupt "
                    f"(likely interrupted save), skipping...[/]"
                    )
            try:
                os.remove(ckpt)
                console.print(f"  [dim]🗑️  Removed corrupt checkpoint: {os.path.basename(ckpt)}[/]")
            except OSError:
                pass
            continue

    return None

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

# ════════════════════════════════════════════════════════════════════════════
# 9a. SETUP HELPER FUNCTIONS (extracted from train)
# ════════════════════════════════════════════════════════════════════════════

def _resolve_device(device_arg: str) -> str:
    """Resolve 'auto' device to the best available hardware."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _seed_everything(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _load_or_build_tokenizer(args, allowed_ops: List[str]) -> BPETokenizer:
    """Load tokenizer from a continued run, or build a fresh one from samples."""
    if args.continue_run is not None:
        tok_dir = args.continue_run
        tok_file = os.path.join(tok_dir, "tokenizer.json")
        if os.path.exists(tok_file):
            tokenizer = BPETokenizer.from_pretrained(tok_dir)
            console.print(f"  [green]✓ Tokenizer restored from {tok_dir}[/]")
            return tokenizer
        else:
            console.print(f"[bold red]❌ No tokenizer.json in {tok_dir}. Cannot continue.[/]")
            sys.exit(1)
    else:
        return build_tokenizer_from_samples(
            n_programs=args.tokenizer_initial_nr,
            allowed_ops=allowed_ops,
            max_params=args.max_params,
            max_ops=args.max_ops,
            param_range=(args.param_min - current_epoch, args.param_max + current_epoch),
            bpe_vocab_size=args.bpe_vocab_size,
            task_type=args.task,
        )

def _resolve_model_config(args, tokenizer: BPETokenizer) -> Tuple[dict, LLVMGPTConfig]:
    if args.continue_run is not None:
        config_path = os.path.join(args.continue_run, "config.json")
        if os.path.exists(config_path):
            model_config = LLVMGPTConfig.from_pretrained(args.continue_run)
            cfg = {
                "d_model": model_config.d_model,
                "n_heads": model_config.n_heads,
                "n_layers": model_config.n_layers,
            }
            console.print(f"  [green]✓ Model config restored from {args.continue_run}[/]")
            return cfg, model_config
        else:
            console.print(f"[bold red]❌ No config.json in {args.continue_run}. Cannot continue.[/]")
            sys.exit(1)

    elif args.d_model > 0 and args.n_layers > 0 and args.n_heads > 0:
        cfg = {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
        }
    else:
        console.print(
            f"[bold cyan]Searching for architecture "
            f"(target ~{args.target_params:,} params)...[/]"
        )
        cfg = find_model_config(tokenizer.vocab_size, args.target_params, args.max_seq_len, ffn=args.ffn)

    ffn_val = args.ffn if args.ffn > 0 else 4 * cfg["d_model"]

    model_config = LLVMGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        ffn=ffn_val,
    )
    return cfg, model_config

def _create_model(model_config: LLVMGPTConfig, device: str) -> TinyClassifierGPT:
    """Instantiate the model and move to device."""
    return TinyClassifierGPT(model_config).to(device)


def _print_model_summary(model: TinyClassifierGPT):
    """Print a Rich panel with the torchinfo model summary."""
    from torchinfo import summary as torchinfo_summary

    model_stats = torchinfo_summary(
        model,
        input_size=(1, args.max_seq_len),
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


def _load_checkpoint_for_continue(args, model: TinyClassifierGPT, device: str) -> Optional[dict]:
    """
    Load checkpoint weights and state for --continue or --resume.
    Returns the checkpoint dict (or None if no resume).
    """
    if args.continue_run is not None:
        run_path = args.continue_run
        if not os.path.isdir(run_path):
            console.print(f"[bold red]❌ Run directory not found: {run_path}[/]")
            sys.exit(1)

        ckpt_file = _find_latest_checkpoint(run_path)
        if ckpt_file is None:
            console.print(f"[bold red]❌ No model_epoch_*.pt checkpoints found in: {run_path}[/]")
            sys.exit(1)

        console.print(f"[bold yellow]🔄 Continuing from: {ckpt_file}[/]")
        checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print("  [green]✓ Model weights loaded[/]")

        # Verify weights differ from random init
        sample_param = next(iter(model.parameters()))
        ckpt_sample = list(checkpoint["model_state_dict"].values())[0]
        if torch.allclose(sample_param.cpu(), ckpt_sample.cpu(), atol=1e-6):
            console.print("  [green]✓ Verified: weights differ from random init[/]")
        else:
            console.print("  [bold red]⚠ WARNING: loaded weights may match random init![/]")

        return checkpoint

    elif args.resume is not None:
        if not os.path.isfile(args.resume):
            console.print(f"[bold red]❌ Checkpoint not found: {args.resume}[/]")
            sys.exit(1)

        console.print(f"[bold yellow]🔄 Resuming from checkpoint: {args.resume}[/]")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print("  [green]✓ Model weights loaded[/]")
        return checkpoint

    return None


def _extract_resumed_state(checkpoint: Optional[dict]) -> dict:
    """
    Extract all resumable state from a checkpoint dict.
    Returns a dict with keys for epoch, losses, plotter state, etc.
    """
    if checkpoint is None:
        return {
            "start_epoch": 0,
            "train_losses": [],
            "val_losses": [],
            "best_val_loss": float("inf"),
            "total_samples": 0,
            "plotter_state": None,
            "replay_state": None,
            "ema_train": None,
            "global_batch": 0,
        }

    return {
        "start_epoch": checkpoint.get("epoch", 0),
        "train_losses": checkpoint.get("train_losses_hist", []),
        "val_losses": checkpoint.get("val_losses_hist", []),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "total_samples": checkpoint.get("total_samples", 0),
        "plotter_state": checkpoint.get("plotter_state", None),
        "replay_state": checkpoint.get("replay_buffer_state", None),
        "ema_train": checkpoint.get("ema_train", None),
        "global_batch": checkpoint.get("global_batch", 0),
    }


def _restore_rng_state(checkpoint: Optional[dict]):
    """Restore Python/NumPy/Torch RNG states from a checkpoint.
    If restoration fails, log a warning and continue with fresh RNG state."""
    if checkpoint is None:
        return
    rng_state = checkpoint.get("rng_state", None)
    if not rng_state:
        return

    try:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])

        # Fix: ensure the torch RNG state is a CPU ByteTensor
        torch_rng = rng_state["torch"]
        if not isinstance(torch_rng, torch.ByteTensor):
            torch_rng = torch_rng.to(device="cpu", dtype=torch.uint8)
        elif torch_rng.device.type != "cpu":
            torch_rng = torch_rng.cpu()
        torch.random.set_rng_state(torch_rng)

        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            cuda_states = rng_state["torch_cuda"]
            cuda_states = [
                s.to(device="cpu", dtype=torch.uint8)
                if not isinstance(s, torch.ByteTensor) or s.device.type != "cpu"
                else s
                for s in cuda_states
            ]
            torch.cuda.set_rng_state_all(cuda_states)

        console.print("  [green]✓ RNG states restored[/]")

    except Exception as e:
        console.print(
            f"  [yellow]⚠ Could not restore RNG state: {e}[/]\n"
            f"  [yellow]  Continuing with default RNG state (training will proceed normally)[/]"
        )

def _create_plotter(args, cfg: dict, actual_params: int, tokenizer: BPETokenizer,
                    device: str, resumed_state: dict) -> LivePlotter:
    """Create and configure the LivePlotter, restoring state if resuming."""
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

    if resumed_state["plotter_state"] is not None:
        plotter.restore_state(resumed_state["plotter_state"])
        if resumed_state["ema_train"] is not None:
            plotter._ema_train = resumed_state["ema_train"]
        if resumed_state["global_batch"] > 0:
            plotter._global_batch = resumed_state["global_batch"]

    return plotter


def _setup_run_logger(args, cfg: dict, model: TinyClassifierGPT, tokenizer: BPETokenizer,
                      actual_params: int) -> Tuple[Optional[RunLogger], Optional[str]]:
    """Set up the run logger and run directory. Returns (logger, run_dir)."""
    global run_dir, _lp_module

    if args.no_run_log:
        run_dir = None
        _lp_module.run_dir = run_dir
        return None, None

    if args.continue_run is not None:
        run_dir = args.continue_run
        _lp_module.run_dir = run_dir
        run_logger = RunLogger(base_dir=args.run_dir, reuse_path=run_dir)
        console.print(f"[bold cyan]📁 Continuing run log in: {run_dir}[/]")
    else:
        run_logger = RunLogger(base_dir=args.run_dir)
        run_dir = run_logger.get_base_dir()
        _lp_module.run_dir = run_dir
        console.print(f"[bold cyan]📁 Run logging to: {run_logger.path}[/]")
        run_logger.log_config(args)
        run_logger.log_model_summary(
            model_name="TinyClassifierGPT",
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

    # ── ALWAYS save tokenizer + config to run_dir ───────────────────────
    # This ensures they're present for future --continue, even if the
    # original files were lost or this is a re-continued run.
    if run_dir is not None:
        tokenizer.save_pretrained(run_dir)
        model.config.save_pretrained(run_dir)
        if args.continue_run is None:
            console.print(f"  [green]✓ Tokenizer + config saved to {run_dir}/ (for --continue)[/]")
        else:
            console.print(f"  [green]✓ Tokenizer + config re-saved to {run_dir}/[/]")

    return run_logger, run_dir

def _setup_csv_logger(run_dir_path: Optional[str]) -> CSVTrainingLogger:
    """Create and configure the CSV training logger."""
    global csv_log
    effective_dir = run_dir_path if run_dir_path else "."
    csv_log = CSVTrainingLogger(output_dir=effective_dir)
    csv_log.set_output_dir(effective_dir)
    return csv_log

def _print_config_table(args, cfg: dict, tokenizer: BPETokenizer,
                        actual_params: int, device: str, allowed_ops: List[str],
                        start_epoch: int, resumed_total_samples: int):
    """Print the Rich configuration summary table."""
    config_table = Table(title="Configuration", box=box.ROUNDED, show_lines=True)
    config_table.add_column("Category", style="bold cyan")
    config_table.add_column("Parameter", style="bold")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model", "d_model", str(cfg["d_model"]))
    config_table.add_row("", "ffn_dim", str(args.ffn if args.ffn > 0 else 4 * cfg["d_model"]))
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

    if args.continue_run is not None:
        config_table.add_row("Resume", "continuing from", args.continue_run)
        config_table.add_row("", "start_epoch", str(start_epoch))
        config_table.add_row("", "resumed samples", f"{resumed_total_samples:,}")

    console.print(config_table)

def _build_optimizer(args, model: TinyClassifierGPT) -> torch.optim.Optimizer:
    opt_cls = OPTIMIZERS[args.optimizer]
    opt_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        opt_kwargs["momentum"] = args.momentum
    return opt_cls(model.parameters(), **opt_kwargs)

def _build_scheduler(args, optimizer, model=None):
    """Create the LR scheduler from args. Returns (scheduler, is_plateau)."""
    if args.scheduler == "warmup_cosine":
        scheduler = build_warmup_cosine(
            optimizer, args.epochs, warmup_epochs=min(5, args.epochs // 5)
        )
    elif args.scheduler == "minima_explorer":
        if model is None:
            raise ValueError("minima_explorer scheduler requires model reference")
        scheduler = AdaptiveLocalMinimaExplorer(
            optimizer=optimizer,
            model=model,
            patience=20,
            max_probes=6,
            exploration_lr_multiplier=10.0,
            perturbation_scale=0.01,
            verbose=True,
        )
        return scheduler, False
    else:
        scheduler = SCHEDULERS[args.scheduler](optimizer, args.epochs)
    is_plateau = args.scheduler == "plateau"
    return scheduler, is_plateau

def _restore_optimizer_and_scheduler(checkpoint: Optional[dict], optimizer, scheduler,
                                     device: str):
    """Restore optimizer and scheduler state dicts from a checkpoint."""
    if checkpoint is None:
        return

    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            console.print("  [green]✓ Optimizer state restored[/]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not restore optimizer state: {e}[/]")

    if "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            console.print("  [green]✓ Scheduler state restored[/]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Could not restore scheduler state: {e}[/]")


def _setup_replay_buffer(args, resumed_state: dict) -> Optional[ReplayBuffer]:
    """Create and optionally restore the replay buffer."""
    if args.replay_save_rate <= 0:
        return None

    replay_buffer = ReplayBuffer(
        save_rate=args.replay_save_rate,
        sprinkle_rate=args.replay_sprinkle_rate,
        max_size=args.replay_max_size,
    )
    console.print(
        f"[bold cyan]🔄 Replay buffer enabled: "
        f"save_rate={args.replay_save_rate:.0%}, "
        f"sprinkle_rate={args.replay_sprinkle_rate:.0%}, "
        f"max_size={args.replay_max_size:,}[/]"
    )

    if resumed_state["replay_state"] is not None:
        replay_buffer.restore_state(resumed_state["replay_state"])

    return replay_buffer


def _validate_allowed_ops(allowed_ops: List[str]):
    """Validate that all specified ops are supported."""
    valid_ops = list(list_supported_ops().keys())
    for op in allowed_ops:
        if op not in valid_ops:
            raise ValueError(f"Invalid op '{op}'. Valid: {valid_ops}")


# ════════════════════════════════════════════════════════════════════════════
# 9b. EPOCH-LEVEL HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _run_train_epoch(
    epoch: int,
    total_epochs: int,
    model: TinyClassifierGPT,
    optimizer,
    args,
    batch_gen_kwargs: dict,
    loss_kwargs: dict,
    tokenizer: BPETokenizer,
    device: str,
    plotter: LivePlotter,
    run_logger: Optional[RunLogger],
    csv_log: CSVTrainingLogger,
    timer: TimeEstimator,
    epoch_ctrl: EpochController,
    total_samples: int,
    scheduler=None,
) -> Tuple[float, float, int]:
    """
    Run one training epoch (classification mode).

    Returns:
        (avg_train_loss, avg_accuracy, updated_total_samples)
    """
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
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
        TextColumn("[dim]acc={task.fields[acc]:.0%}[/]"),
        TextColumn("•"),
        TextColumn("[dim]eta={task.fields[eta]}[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "train", total=args.batches_per_epoch,
            loss=0.0, ema=0.0, acc=0.0, eta="...",
        )

        ema_loss = None

        for batch_idx in range(args.batches_per_epoch):
            csv_log.start_batch_timer()

            prepared = _prepare_batch(**batch_gen_kwargs)
            if prepared is None:
                progress.update(task, advance=1, loss=0.0,
                                ema=ema_loss or 0.0, acc=0.0,
                                eta=timer.eta(epoch - 1, epoch_ctrl.epochs))
                continue

            batch, inp, labels = prepared
            total_samples += len(batch)

            loss, loss_val, accuracy = _compute_batch_loss(
                model, inp, labels, **loss_kwargs,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            optimizer.step()

            if isinstance(scheduler, AdaptiveLocalMinimaExplorer):
                scheduler.report_loss(loss.item())

            bl = loss.item()
            epoch_loss += bl
            epoch_accuracy += accuracy
            n_batches += 1

            if ema_loss is None:
                ema_loss = bl
            else:
                ema_loss = 0.05 * bl + 0.95 * ema_loss

            current_lr = optimizer.param_groups[0]["lr"]
            preds = get_batch_predictions(model, tokenizer, batch, device, task=args.task)

            csv_log.log_train_batch(
                epoch=epoch, batch_idx=batch_idx,
                total_loss=bl, ce_loss=bl,
                value_loss=0.0, structure_loss=0.0,
                parse_rate=accuracy, predictions=preds,
                lr=current_lr,
                grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                n_samples_in_batch=len(batch),
            )

            plotter.update_batch(bl, model=model)
            plotter.update_topo(model, inp)
            plotter.update_jacobi_fields(model, inp, tokenizer)

            if preds:
                plotter.accumulate_predictions(preds)
                plotter.update_prediction_diffs(preds)

            if run_logger:
                run_logger.log_batch_loss_train(epoch, batch_idx, bl, ema_loss)

            progress.update(
                task, advance=1, loss=bl, ema=ema_loss,
                acc=accuracy,
                eta=timer.eta(epoch - 1, epoch_ctrl.epochs),
            )

    avg_train_loss = epoch_loss / max(n_batches, 1)
    avg_accuracy = epoch_accuracy / max(n_batches, 1)
    return avg_train_loss, avg_accuracy, total_samples

def _run_val_epoch(
    epoch: int,
    total_epochs: int,
    model: TinyClassifierGPT,
    args,
    batch_gen_kwargs: dict,
    loss_kwargs: dict,
    tokenizer: BPETokenizer,
    device: str,
    optimizer,
    plotter: LivePlotter,
    run_logger: Optional[RunLogger],
    csv_log: CSVTrainingLogger,
) -> float:
    """
    Run one validation epoch (classification mode).

    Returns:
        avg_val_loss
    """
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
        TextColumn("•"),
        TextColumn("[dim]acc={task.fields[acc]:.0%}[/]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("val", total=args.val_batches, loss=0.0, acc=0.0)

        with torch.no_grad():
            for val_batch_idx in range(args.val_batches):
                csv_log.start_batch_timer()

                prepared = _prepare_batch(**batch_gen_kwargs)
                if prepared is None:
                    progress.update(task, advance=1, loss=0.0, acc=0.0)
                    continue

                batch, inp, labels = prepared

                vl_loss, vl_scalar, vl_accuracy = _compute_batch_loss(
                    model, inp, labels, **loss_kwargs,
                )

                vl = vl_loss.item()
                val_loss += vl
                val_batches += 1

                preds = get_batch_predictions(model, tokenizer, batch, device, task=args.task)

                csv_log.log_val_batch(
                    epoch=epoch, batch_idx=val_batch_idx,
                    total_loss=vl, ce_loss=vl,
                    value_loss=0.0, structure_loss=0.0,
                    parse_rate=vl_accuracy, predictions=preds,
                    lr=optimizer.param_groups[0]["lr"],
                )

                plotter.update_val_batch(vl)

                if val_batch_idx == 0 or val_batch_idx == args.val_batches - 1:
                    preds = get_batch_predictions(model, tokenizer, batch, device, task=args.task)
                    if preds:
                        plotter.update_predictions(preds)
                        plotter.update_prediction_diffs(preds)

                if run_logger:
                    run_logger.log_batch_loss_val(epoch, val_batches, vl)

                progress.update(task, advance=1, loss=vl, acc=vl_accuracy)

    plotter.finish_val_epoch()
    return val_loss / max(val_batches, 1)

def _step_scheduler(scheduler, is_plateau: bool, avg_val_loss: float):
    """Advance the LR scheduler by one step."""
    if isinstance(scheduler, AdaptiveLocalMinimaExplorer):
        return  # MinimaExplorer manages its own LR via report_loss()
    if is_plateau:
        scheduler.step(avg_val_loss)
    else:
        scheduler.step()

def _print_epoch_summary(
    epoch: int,
    total_epochs: int,
    avg_train_loss: float,
    avg_val_loss: float,
    current_lr: float,
    total_samples: int,
    elapsed: float,
    timer: TimeEstimator,
    is_best: bool,
    avg_accuracy: float = 0.0,
):
    """Print the Rich epoch summary panel."""
    eta_str = timer.eta(epoch, total_epochs)
    elapsed_str = timer.elapsed_total()
    best_marker = " [bold green]★ best[/]" if is_best else ""

    epoch_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    for _ in range(14):
        epoch_table.add_column()

    epoch_table.add_row(
        "[bold]train:[/]", f"[cyan]{avg_train_loss:.4f}[/]",
        "[bold]val:[/]", f"[magenta]{avg_val_loss:.4f}[/]",
        "[bold]acc:[/]", f"[green]{avg_accuracy:.1%}[/]",
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
            width=min(console.width, 130),
        )
    )

def _log_epoch_to_run_logger(
        run_logger: Optional[RunLogger],
        epoch: int,
        avg_train_loss: float,
        avg_val_loss: float,
        current_lr: float,
        elapsed: float,
        total_samples: int,
        is_best: bool,
        model: TinyClassifierGPT,
        tokenizer: BPETokenizer,
        device: str,
        args,
        allowed_ops: List[str],
):
    """Log epoch summary and example samples to the run logger."""
    if run_logger is None:
        return

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

    # ── FIX: Sample from the CORE of the trained distribution ───────────
    # The training param_range expands as (param_min - epoch, param_max + epoch),
    # but sampling uniformly from that full range at high epochs produces
    # mostly out-of-distribution inputs where the model immediately emits EOS.
    # Instead, sample from the inner ~50% of the current range so the logged
    # samples reflect what the model has actually learned well.
    half_expansion = current_epoch // 2
    sample_param_range = (
        args.param_min - half_expansion,
        args.param_max + half_expansion,
    )

    example_samples = generate_example_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=n_to_log if n_to_log else 999999,
        max_params=args.max_params,
        max_ops=args.max_ops,
        allowed_ops=allowed_ops,
        param_range=sample_param_range,
        task=args.task,
        max_turnstiles=args.max_turnstiles,
    )

    run_logger.log_samples(epoch, example_samples, n_samples=n_to_log)
    run_logger.flush_losses()

def _do_checkpointing(
        epoch: int,
        model: TinyClassifierGPT,
        optimizer,
        scheduler,
        avg_train_loss: float,
        avg_val_loss: float,
        best_val_loss: float,
        train_losses_hist: List[float],
        val_losses_hist: List[float],
        total_samples: int,
        plotter: LivePlotter,
        replay_buffer: Optional['ReplayBuffer'],
        tokenizer: BPETokenizer,
        save_path: str,
        is_best: bool,
        args,
):
    """Save epoch checkpoint, prune old ones, and save best model if needed."""
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
        train_losses_hist=train_losses_hist,
        val_losses_hist=val_losses_hist,
        total_samples=total_samples,
        plotter_state=plotter.get_state() if plotter.enabled else None,
        replay_buffer_state=replay_buffer.get_state() if replay_buffer else None,
        args_dict=vars(args),
        ema_train=plotter._ema_train if plotter.enabled else None,
        global_batch=plotter._global_batch if plotter.enabled else 0,
    )
    console.print(f"  [dim]💾 Saved checkpoint: {ckpt_path}[/]")

    _prune_checkpoints(save_path, max_keep=10)

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
            train_losses_hist=train_losses_hist,
            val_losses_hist=val_losses_hist,
            total_samples=total_samples,
            plotter_state=plotter.get_state() if plotter.enabled else None,
            replay_buffer_state=replay_buffer.get_state() if replay_buffer else None,
            args_dict=vars(args),
            ema_train=plotter._ema_train if plotter.enabled else None,
            global_batch=plotter._global_batch if plotter.enabled else 0,
        )

        best_hf_path = f"{save_path}/best"
        model.save_pretrained(best_hf_path)
        tokenizer.save_pretrained(best_hf_path)
        console.print(
            f"  [bold green]⭐ New best model saved: {best_path} "
            f"(val_loss={avg_val_loss:.4f})[/]"
        )


def _save_final_model(
        model: TinyClassifierGPT,
        tokenizer: BPETokenizer,
        save_path: str,
        train_losses_hist: List[float],
        val_losses_hist: List[float],
        best_val_loss: float,
        total_samples: int,
        actual_params: int,
        epoch: int,
        timer: TimeEstimator,
        args,
):
    """Save the final model, tokenizer, and training metadata."""
    if not save_path:
        return

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

def _finalize_training(
        epoch_ctrl: EpochController,
        csv_log: CSVTrainingLogger,
        plotter: LivePlotter,
        run_dir_path: Optional[str],
):
    """Stop controllers, close loggers, finalize plots."""
    epoch_ctrl.stop()

    csv_log.close()
    console.print(
        f"[bold green]📊 CSV logs saved to: "
        f"{os.path.join(run_dir_path if run_dir_path else '.', 'batch_log.csv')} "
        f"and epoch_log.csv[/]"
    )

    # Save the final plot to file, then close the window automatically
    if plotter.enabled:
        plotter._save_to_file()
        console.print(f"[bold green]📊 Final plot saved to: {plotter.plot_file}[/]")
        plt.close('all')

# ════════════════════════════════════════════════════════════════════════════
# 9c. REFACTORED TRAIN — the coordinator
# ════════════════════════════════════════════════════════════════════════════

def _compute_total_epochs(args, resumed_start_epoch: int) -> int:
    """
    When continuing a run, add the requested epochs ON TOP of the
    already-completed ones so the model trains for the same number
    of NEW epochs as a fresh run would.

    Fresh run:    --epochs 30  →  trains epochs 1..30
    Continue:     --epochs 30, checkpoint at epoch 30  →  trains epochs 31..60
    """
    if args.continue_run is not None and resumed_start_epoch > 0:
        return resumed_start_epoch + args.epochs
    return args.epochs

def train(args: argparse.Namespace):
    global current_epoch

    # ── Parse & validate ────────────────────────────────────────────────
    allowed_ops = [op.strip() for op in args.allowed_ops.split(",")]
    _validate_allowed_ops(allowed_ops)

    device = _resolve_device(args.device)
    _seed_everything(args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer = _load_or_build_tokenizer(args, allowed_ops)

    # ── Model config & creation ─────────────────────────────────────────
    cfg, model_config = _resolve_model_config(args, tokenizer)
    model = _create_model(model_config, device)
    _print_model_summary(model)
    actual_params = model.count_parameters()

    # ── Load checkpoint (--continue / --resume) ─────────────────────────
    checkpoint = _load_checkpoint_for_continue(args, model, device)
    resumed = _extract_resumed_state(checkpoint)
    _restore_rng_state(checkpoint)

    # ── Compute effective total epochs ──────────────────────────────────
    effective_total_epochs = _compute_total_epochs(args, resumed["start_epoch"])

    # ── Plotter (open immediately) ──────────────────────────────────────
    plotter = _create_plotter(args, cfg, actual_params, tokenizer, device, resumed)

    # ── Run logger & CSV logger ─────────────────────────────────────────
    run_logger, run_dir_path = _setup_run_logger(args, cfg, model, tokenizer, actual_params)
    csv_log = _setup_csv_logger(run_dir_path)

    # ── Config summary ──────────────────────────────────────────────────
    _print_config_table(args, cfg, tokenizer, actual_params, device,
                        allowed_ops, resumed["start_epoch"],
                        resumed["total_samples"])

    # ── Optimizer & scheduler ───────────────────────────────────────────
    optimizer = _build_optimizer(args, model)
    scheduler, is_plateau = _build_scheduler(args, optimizer, model=model)
    _restore_optimizer_and_scheduler(checkpoint, optimizer, scheduler, device)

    # ── Epoch controller & timer ────────────────────────────────────────
    #    Use effective_total_epochs so --continue adds NEW epochs on top
    epoch_ctrl = EpochController(initial_epochs=effective_total_epochs, step=10, plotter=plotter)
    epoch_ctrl.start()
    timer = TimeEstimator()

    # ── Replay buffer ───────────────────────────────────────────────────
    replay_buffer = _setup_replay_buffer(args, resumed)

    # ── Shared kwargs for batch generation & loss ───────────────────────
    batch_gen_kwargs = dict(
        tokenizer=tokenizer, batch_size=args.batch_size,
        max_params=args.max_params, max_ops=args.max_ops,
        allowed_ops=allowed_ops,
        param_range=(args.param_min - current_epoch, args.param_max + current_epoch),
        max_seq_len=args.max_seq_len, device=device,
        replay_buffer=replay_buffer,
        task=args.task,
        max_turnstiles=args.max_turnstiles,
    )
    loss_kwargs = dict(
        tokenizer=tokenizer, device=device,
        value_loss_alpha=args.value_loss_alpha,
        structure_loss_alpha=args.structure_loss_alpha,
        length_loss_alpha=args.length_loss_alpha,
    )

    # ── Simplify loss for turnstile task (binary 0/1 output) ────────────
    if args.task == "turnstile":
        loss_kwargs["structure_loss_alpha"] = 0.0
        loss_kwargs["length_loss_alpha"] = 0.0
        loss_kwargs["value_loss_alpha"] = 0.0
        # ── Swap prediction error scorer for binary turnstile scoring ───
        _lp_module._prediction_error_score = _prediction_error_score_turnstile
    else:
        # Ensure default scorer is active (matters if module was reused)
        _lp_module._prediction_error_score = _prediction_error_score

    # ── State variables ─────────────────────────────────────────────────
    best_val_loss = resumed["best_val_loss"]
    train_losses_hist = list(resumed["train_losses"])
    val_losses_hist = list(resumed["val_losses"])
    total_samples = resumed["total_samples"]
    save_path = os.path.normpath(run_dir_path) if run_dir_path else "llvm_gpt_model"

    # ── Banner ──────────────────────────────────────────────────────────
    new_epochs = effective_total_epochs - resumed["start_epoch"]
    banner_text = (
        f"[bold white]Training for {new_epochs} epochs "
        f"({resumed['start_epoch']+1}→{effective_total_epochs})  │  "
        f"{args.batches_per_epoch} batches/epoch  │  "
        f"batch_size={args.batch_size}  │  "
        f"Press +/- to adjust epochs, q to stop[/]"
    )
    if resumed["start_epoch"] > 0:
        banner_text += (
            f"\n[bold yellow]Resuming from epoch {resumed['start_epoch']} — "
            f"will run {new_epochs} new epochs (total target: {effective_total_epochs})[/]"
        )
    console.print(Panel(
        banner_text,
        title="[bold green]🚀 Starting Training",
        border_style="green",
    ))

    # ════════════════════════════════════════════════════════════════════
    # EPOCH LOOP
    # ════════════════════════════════════════════════════════════════════
    epoch = resumed["start_epoch"]
    while True:
        current_epoch = epoch
        epoch += 1

        batch_gen_kwargs["param_range"] = (
            args.param_min - current_epoch,
            args.param_max + current_epoch,
        )

        total_epochs = epoch_ctrl.epochs
        if epoch > total_epochs:
            break
        epoch_ctrl.set_min(epoch)
        t0 = time.time()

        # ── Train ───────────────────────────────────────────────────────
        avg_train_loss, avg_accuracy, total_samples = _run_train_epoch(
            epoch, total_epochs, model, optimizer, args,
            batch_gen_kwargs, loss_kwargs, tokenizer, device,
            plotter, run_logger, csv_log, timer, epoch_ctrl, total_samples,
            scheduler=scheduler,
        )

        # ── Validate ────────────────────────────────────────────────────
        avg_val_loss = _run_val_epoch(
            epoch, total_epochs, model, args,
            batch_gen_kwargs, loss_kwargs, tokenizer, device,
            optimizer, plotter, run_logger, csv_log,
        )

        # ── Scheduler step ──────────────────────────────────────────────
        _step_scheduler(scheduler, is_plateau, avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        timer.add_epoch(elapsed)

        # ── Track best ──────────────────────────────────────────────────
        train_losses_hist.append(avg_train_loss)
        val_losses_hist.append(avg_val_loss)
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        csv_log.end_epoch(epoch=epoch, lr=current_lr, epoch_time_sec=elapsed)

        # ── Epoch summary ───────────────────────────────────────────────
        _print_epoch_summary(epoch, total_epochs, avg_train_loss, avg_val_loss,
                             current_lr, total_samples, elapsed, timer, is_best,
                             avg_accuracy=avg_accuracy)

        # ── Logging ─────────────────────────────────────────────────────
        _log_epoch_to_run_logger(
            run_logger, epoch, avg_train_loss, avg_val_loss, current_lr,
            elapsed, total_samples, is_best, model, tokenizer, device,
            args, allowed_ops,
        )

        # ── Plot update ─────────────────────────────────────────────────
        plotter.update_epoch(avg_train_loss, avg_val_loss, current_lr, model=model)

        # ── Checkpointing ───────────────────────────────────────────────
        _do_checkpointing(
            epoch, model, optimizer, scheduler,
            avg_train_loss, avg_val_loss, best_val_loss,
            train_losses_hist, val_losses_hist, total_samples,
            plotter, replay_buffer, tokenizer, save_path, is_best, args,
        )

        # ── Sync to remote ─────────────────────────────────────────────
        _maybe_run_sync(run_dir_path, args.sync)

        # ── Graceful stop ───────────────────────────────────────────────
        if _interrupt_count >= 1:
            console.print("[bold yellow]Graceful stop after epoch.[/]")
            break

    # ════════════════════════════════════════════════════════════════════
    # FINALIZE
    # ════════════════════════════════════════════════════════════════════
    _finalize_training(epoch_ctrl, csv_log, plotter, run_dir_path)
    _save_final_model(model, tokenizer, save_path, train_losses_hist,
                      val_losses_hist, best_val_loss, total_samples,
                      actual_params, epoch, timer, args)

    # ── Final sync (run once more, then wait for it to finish) ──────
    _maybe_run_sync(run_dir_path, args.sync)
    _wait_for_sync()

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════
    total_runtime = time.time() - timer.start_time

    original_print("\n" + "=" * 60)
    original_print(f"LOSS: {train_losses_hist[-1]:.6f}")
    original_print(f"VAL_LOSS: {val_losses_hist[-1]:.6f}")
    original_print(f"BEST_VAL_LOSS: {best_val_loss:.6f}")
    original_print(f"TOTAL_SAMPLES: {total_samples}")
    original_print(f"PARAMS: {actual_params}")
    original_print(f"LR_FINAL: {optimizer.param_groups[0]['lr']:.2e}")
    original_print(f"RUNTIME: {total_runtime:.1f}")
    original_print("=" * 60 + "\n")

    return model, tokenizer

# ════════════════════════════════════════════════════════════════════════════
# 11.  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

# From train.py — existing but incomplete
def compute_jacobian_spectra(model, input_ids, device):
    """Compute singular values of the per-layer Jacobian via finite differences."""
    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = output.hidden_states

    if hidden_states is None:
        return []

    spectra = []
    for ell in range(len(hidden_states) - 1):
        h_in = hidden_states[ell][0].detach().float()   # (T, D)
        h_out = hidden_states[ell + 1][0].detach().float()  # (T, D)
        delta = h_out - h_in

        h_in_c = h_in - h_in.mean(0)
        delta_c = delta - delta.mean(0)

        try:
            J = torch.linalg.lstsq(h_in_c, delta_c).solution  # (D, D)
            svs = torch.linalg.svdvals(J).cpu().numpy()
        except Exception:
            svs = np.zeros(h_in.shape[-1])
        spectra.append(svs)

    return spectra

def compute_jacobian_invariants(spectra_or_jacobians):
    """From per-layer Jacobians, extract div/curl/shear per token."""
    results = []
    for J in spectra_or_jacobians:  # ← was `jacobians` (undefined)
        # J should be a 2D tensor (D, D)
        if isinstance(J, np.ndarray):
            J = torch.from_numpy(J).float()
        D = J.shape[0]
        div_val = torch.trace(J).item()
        J_antisym = (J - J.T) / 2
        curl_val = torch.norm(J_antisym, p='fro').item()
        J_sym = (J + J.T) / 2
        shear_mat = J_sym - (torch.trace(J_sym) / D) * torch.eye(D, device=J.device)
        shear_val = torch.norm(shear_mat, p='fro').item()
        results.append({'div': div_val, 'curl': curl_val, 'shear': shear_val})
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

# ════════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER — save & sprinkle back generated programs
# ════════════════════════════════════════════════════════════════════════════

def generate_single_sample_turnstile(
    tokenizer: BPETokenizer,
    max_turnstiles: int = 10,
    max_seq_len: int = 2048,
    bias_equal: float = 0.5,
) -> Optional[Tuple[List[int], int]]:
    """Generate a single turnstile task sample, tokenized and ready for training."""

    try:
        prompt, result = generate_turnstile_function(
            max_turnstiles=max_turnstiles,
            bias_equal=bias_equal,
        )
    except Exception:
        return None

    result_str = str(result)  # "0" or "1"
    text = f"{prompt}{result_str}"

    # Same encoding logic as generate_single_sample for infix
    full_ids = tokenizer.encode(text)
    prompt_only_ids = tokenizer.encode(prompt)

    # Find longest common prefix
    common_len = 0
    for a, b in zip(prompt_only_ids, full_ids):
        if a == b:
            common_len += 1
        else:
            break
    else:
        common_len = len(prompt_only_ids)

    prompt_len = 1 + common_len  # +1 for <bos>

    answer_ids = full_ids[common_len:]
    if not answer_ids:
        # Fallback: same as infix version
        accumulated = ""
        prompt_len_fallback = 1
        for i, tid in enumerate(full_ids):
            accumulated = tokenizer.decode(full_ids[:i + 1])
            if accumulated.rstrip().endswith(" "):
                remaining = tokenizer.decode(full_ids[i + 1:]).strip()
                if remaining == result_str or remaining.startswith(result_str):
                    prompt_len_fallback = 1 + (i + 1)
                    break
        prompt_len = prompt_len_fallback

        total_len = 1 + len(full_ids) + 1
        if prompt_len >= total_len - 1:
            return None

    ids = (
        [tokenizer.bos_token_id]
        + full_ids
        + [tokenizer.eos_token_id]
    )

    return ids, prompt_len

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

    if not torch.cuda.is_available():
        console.print("\n[bold orange]No GPU detected.[/]")

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

