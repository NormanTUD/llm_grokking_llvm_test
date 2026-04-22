# csv_logger.py

"""
Comprehensive CSV logger for LLVM IR GPT training.
Produces two CSV files:
  1. batch_log.csv  — one row per batch (train + val)
  2. epoch_log.csv  — one row per epoch with aggregated stats

Usage:
    from csv_logger import CSVTrainingLogger
    csv_log = CSVTrainingLogger(output_dir="runs/my_run")
    # ... inside training loop ...
    csv_log.log_train_batch(epoch, batch_idx, metrics_dict)
    csv_log.log_val_batch(epoch, batch_idx, metrics_dict)
    csv_log.end_epoch(epoch, lr, elapsed_secs)
    csv_log.close()
"""

import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _EpochAccumulator:
    """Accumulates per-batch stats within a single epoch."""
    # Train
    train_losses: List[float] = field(default_factory=list)
    train_ce_losses: List[float] = field(default_factory=list)
    train_value_losses: List[float] = field(default_factory=list)
    train_structure_losses: List[float] = field(default_factory=list)
    train_parse_rates: List[float] = field(default_factory=list)

    # Per-prediction tracking (train)
    train_correct: int = 0
    train_wrong: int = 0
    train_unparseable: int = 0
    train_abs_diffs: List[float] = field(default_factory=list)
    train_signed_diffs: List[float] = field(default_factory=list)
    train_structure_penalties: List[float] = field(default_factory=list)

    # Val
    val_losses: List[float] = field(default_factory=list)
    val_ce_losses: List[float] = field(default_factory=list)
    val_value_losses: List[float] = field(default_factory=list)
    val_structure_losses: List[float] = field(default_factory=list)
    val_parse_rates: List[float] = field(default_factory=list)

    # Per-prediction tracking (val)
    val_correct: int = 0
    val_wrong: int = 0
    val_unparseable: int = 0
    val_abs_diffs: List[float] = field(default_factory=list)
    val_signed_diffs: List[float] = field(default_factory=list)
    val_structure_penalties: List[float] = field(default_factory=list)

    # Timing
    batch_times_train: List[float] = field(default_factory=list)
    batch_times_val: List[float] = field(default_factory=list)

    # Gradient norms (train only)
    grad_norms: List[float] = field(default_factory=list)

    def reset(self):
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if isinstance(val, list):
                val.clear()
            elif isinstance(val, int):
                setattr(self, f, 0)


class CSVTrainingLogger:
    """
    Writes two CSV files with exhaustive training metrics.

    batch_log.csv columns:
        timestamp, epoch, batch_idx, phase (train/val),
        total_loss, ce_loss, value_loss, structure_loss, parse_rate,
        n_correct, n_wrong, n_unparseable,
        mean_abs_diff, median_abs_diff, min_abs_diff, max_abs_diff,
        mean_signed_diff, std_abs_diff,
        mean_structure_penalty, batch_time_sec, grad_norm, lr

    epoch_log.csv columns:
        timestamp, epoch,
        train_loss_mean, train_loss_std, train_loss_min, train_loss_max,
        train_ce_loss_mean, train_value_loss_mean, train_structure_loss_mean,
        train_parse_rate_mean,
        train_correct, train_wrong, train_unparseable, train_accuracy_pct,
        train_mean_abs_diff, train_median_abs_diff, train_min_abs_diff, train_max_abs_diff,
        train_std_abs_diff, train_mean_signed_diff,
        train_mean_structure_penalty, train_median_structure_penalty,
        val_loss_mean, val_loss_std, val_loss_min, val_loss_max,
        val_ce_loss_mean, val_value_loss_mean, val_structure_loss_mean,
        val_parse_rate_mean,
        val_correct, val_wrong, val_unparseable, val_accuracy_pct,
        val_mean_abs_diff, val_median_abs_diff, val_min_abs_diff, val_max_abs_diff,
        val_std_abs_diff, val_mean_signed_diff,
        val_mean_structure_penalty, val_median_structure_penalty,
        lr, epoch_time_sec,
        cumulative_train_samples, cumulative_wall_time_sec,
        best_val_loss, is_best,
        mean_grad_norm, max_grad_norm,
        train_pct_off_by_0, train_pct_off_by_le_1, train_pct_off_by_le_5,
        train_pct_off_by_le_10, train_pct_off_by_le_100,
        val_pct_off_by_0, val_pct_off_by_le_1, val_pct_off_by_le_5,
        val_pct_off_by_le_10, val_pct_off_by_le_100
    """

    BATCH_COLUMNS = [
        "timestamp", "epoch", "batch_idx", "phase",
        "total_loss", "ce_loss", "value_loss", "structure_loss", "parse_rate",
        "n_correct", "n_wrong", "n_unparseable",
        "mean_abs_diff", "median_abs_diff", "min_abs_diff", "max_abs_diff",
        "mean_signed_diff", "std_abs_diff",
        "mean_structure_penalty",
        "batch_time_sec", "grad_norm", "lr",
    ]

    EPOCH_COLUMNS = [
        "timestamp", "epoch",
        # ── Train loss stats ──
        "train_loss_mean", "train_loss_std", "train_loss_min", "train_loss_max",
        "train_ce_loss_mean", "train_value_loss_mean", "train_structure_loss_mean",
        "train_parse_rate_mean",
        # ── Train prediction stats ──
        "train_correct", "train_wrong", "train_unparseable", "train_accuracy_pct",
        "train_mean_abs_diff", "train_median_abs_diff",
        "train_min_abs_diff", "train_max_abs_diff",
        "train_std_abs_diff", "train_mean_signed_diff",
        "train_mean_structure_penalty", "train_median_structure_penalty",
        # ── Val loss stats ──
        "val_loss_mean", "val_loss_std", "val_loss_min", "val_loss_max",
        "val_ce_loss_mean", "val_value_loss_mean", "val_structure_loss_mean",
        "val_parse_rate_mean",
        # ── Val prediction stats ──
        "val_correct", "val_wrong", "val_unparseable", "val_accuracy_pct",
        "val_mean_abs_diff", "val_median_abs_diff",
        "val_min_abs_diff", "val_max_abs_diff",
        "val_std_abs_diff", "val_mean_signed_diff",
        "val_mean_structure_penalty", "val_median_structure_penalty",
        # ── Global ──
        "lr", "epoch_time_sec",
        "cumulative_train_samples", "cumulative_wall_time_sec",
        "best_val_loss", "is_best",
        "mean_grad_norm", "max_grad_norm",
        # ── Bucket analysis (train) ──
        "train_pct_off_by_0", "train_pct_off_by_le_1",
        "train_pct_off_by_le_5", "train_pct_off_by_le_10",
        "train_pct_off_by_le_100",
        # ── Bucket analysis (val) ──
        "val_pct_off_by_0", "val_pct_off_by_le_1",
        "val_pct_off_by_le_5", "val_pct_off_by_le_10",
        "val_pct_off_by_le_100",
    ]

    def __init__(self, output_dir: str = "."):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._start_wall = time.time()
        self._cumulative_samples = 0
        self._best_val_loss = float("inf")

        # Open files and write headers
        self._batch_path = os.path.join(output_dir, "batch_log.csv")
        self._epoch_path = os.path.join(output_dir, "epoch_log.csv")

        self._batch_file = open(self._batch_path, "w", newline="")
        self._epoch_file = open(self._epoch_path, "w", newline="")

        self._batch_writer = csv.DictWriter(
            self._batch_file, fieldnames=self.BATCH_COLUMNS
        )
        self._epoch_writer = csv.DictWriter(
            self._epoch_file, fieldnames=self.EPOCH_COLUMNS
        )

        self._batch_writer.writeheader()
        self._epoch_writer.writeheader()

        self._batch_file.flush()
        self._epoch_file.flush()

        self._acc = _EpochAccumulator()
        self._batch_timer: Optional[float] = None

    # ════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════

    def set_output_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

    @staticmethod
    def _safe_mean(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    @staticmethod
    def _safe_median(lst: List[float]) -> float:
        if not lst:
            return 0.0
        s = sorted(lst)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return s[mid]

    @staticmethod
    def _safe_std(lst: List[float]) -> float:
        if len(lst) < 2:
            return 0.0
        mean = sum(lst) / len(lst)
        var = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
        return math.sqrt(var)

    @staticmethod
    def _safe_min(lst: List[float]) -> float:
        return min(lst) if lst else 0.0

    @staticmethod
    def _safe_max(lst: List[float]) -> float:
        return max(lst) if lst else 0.0

    @staticmethod
    def _bucket_pcts(abs_diffs: List[float]) -> Dict[str, float]:
        """Compute percentage of predictions within various error buckets."""
        n = len(abs_diffs)
        if n == 0:
            return {
                "pct_off_by_0": 0.0,
                "pct_off_by_le_1": 0.0,
                "pct_off_by_le_5": 0.0,
                "pct_off_by_le_10": 0.0,
                "pct_off_by_le_100": 0.0,
            }
        return {
            "pct_off_by_0": 100.0 * sum(1 for d in abs_diffs if d == 0) / n,
            "pct_off_by_le_1": 100.0 * sum(1 for d in abs_diffs if d <= 1) / n,
            "pct_off_by_le_5": 100.0 * sum(1 for d in abs_diffs if d <= 5) / n,
            "pct_off_by_le_10": 100.0 * sum(1 for d in abs_diffs if d <= 10) / n,
            "pct_off_by_le_100": 100.0 * sum(1 for d in abs_diffs if d <= 100) / n,
        }

    @staticmethod
    def _score_predictions(
        predictions: List[Tuple[str, str, bool]],
    ) -> Tuple[int, int, int, List[float], List[float], List[float]]:
        """
        Score a list of (expected, predicted, _) tuples.

        Returns:
            (n_correct, n_wrong, n_unparseable,
             abs_diffs, signed_diffs, structure_penalties)
        """
        n_correct = 0
        n_wrong = 0
        n_unparseable = 0
        abs_diffs: List[float] = []
        signed_diffs: List[float] = []
        structure_penalties: List[float] = []

        for expected, predicted, _ in predictions:
            try:
                exp_val = int(expected.strip())
            except (ValueError, TypeError, AttributeError):
                n_unparseable += 1
                structure_penalties.append(6.0)
                continue

            pred_cleaned = "".join(
                c for c in predicted if c.isascii() and c.isprintable()
            ).strip()

            try:
                pred_val = int(pred_cleaned)
            except (ValueError, TypeError):
                n_unparseable += 1
                structure_penalties.append(5.0)
                continue

            diff = abs(exp_val - pred_val)
            abs_diffs.append(float(diff))
            signed_diffs.append(float(pred_val - exp_val))

            if diff == 0:
                n_correct += 1
                structure_penalties.append(0.0)
            else:
                n_wrong += 1
                # Approximate structure penalty for CSV (simplified)
                penalty = min(1.0, 0.9 * math.tanh(0.1 * diff))
                structure_penalties.append(penalty)

        return (n_correct, n_wrong, n_unparseable,
                abs_diffs, signed_diffs, structure_penalties)

    # ════════════════════════════════════════════════════════════════════
    # Public API — call these from the training loop
    # ════════════════════════════════════════════════════════════════════

    def start_batch_timer(self):
        """Call at the very start of a batch."""
        self._batch_timer = time.time()

    def log_train_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        ce_loss: float,
        value_loss: float,
        structure_loss: float,
        parse_rate: float,
        predictions: List[Tuple[str, str, bool]],
        lr: float,
        grad_norm: float = 0.0,
        n_samples_in_batch: int = 0,
    ):
        """Log one training batch."""
        batch_time = time.time() - self._batch_timer if self._batch_timer else 0.0
        self._cumulative_samples += n_samples_in_batch

        # Score predictions
        nc, nw, nu, ad, sd, sp = self._score_predictions(predictions)

        # Accumulate for epoch summary
        acc = self._acc
        acc.train_losses.append(total_loss)
        acc.train_ce_losses.append(ce_loss)
        acc.train_value_losses.append(value_loss)
        acc.train_structure_losses.append(structure_loss)
        acc.train_parse_rates.append(parse_rate)
        acc.train_correct += nc
        acc.train_wrong += nw
        acc.train_unparseable += nu
        acc.train_abs_diffs.extend(ad)
        acc.train_signed_diffs.extend(sd)
        acc.train_structure_penalties.extend(sp)
        acc.batch_times_train.append(batch_time)
        acc.grad_norms.append(grad_norm)

        # Write batch row
        row = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "phase": "train",
            "total_loss": f"{total_loss:.6f}",
            "ce_loss": f"{ce_loss:.6f}",
            "value_loss": f"{value_loss:.6f}",
            "structure_loss": f"{structure_loss:.6f}",
            "parse_rate": f"{parse_rate:.4f}",
            "n_correct": nc,
            "n_wrong": nw,
            "n_unparseable": nu,
            "mean_abs_diff": f"{self._safe_mean(ad):.4f}" if ad else "",
            "median_abs_diff": f"{self._safe_median(ad):.4f}" if ad else "",
            "min_abs_diff": f"{self._safe_min(ad):.4f}" if ad else "",
            "max_abs_diff": f"{self._safe_max(ad):.4f}" if ad else "",
            "mean_signed_diff": f"{self._safe_mean(sd):.4f}" if sd else "",
            "std_abs_diff": f"{self._safe_std(ad):.4f}" if ad else "",
            "mean_structure_penalty": f"{self._safe_mean(sp):.4f}" if sp else "",
            "batch_time_sec": f"{batch_time:.4f}",
            "grad_norm": f"{grad_norm:.6f}",
            "lr": f"{lr:.8f}",
        }
        self._batch_writer.writerow(row)
        self._batch_file.flush()

    def log_val_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_loss: float,
        ce_loss: float,
        value_loss: float,
        structure_loss: float,
        parse_rate: float,
        predictions: List[Tuple[str, str, bool]],
        lr: float,
    ):
        """Log one validation batch."""
        batch_time = time.time() - self._batch_timer if self._batch_timer else 0.0

        nc, nw, nu, ad, sd, sp = self._score_predictions(predictions)

        acc = self._acc
        acc.val_losses.append(total_loss)
        acc.val_ce_losses.append(ce_loss)
        acc.val_value_losses.append(value_loss)
        acc.val_structure_losses.append(structure_loss)
        acc.val_parse_rates.append(parse_rate)
        acc.val_correct += nc
        acc.val_wrong += nw
        acc.val_unparseable += nu
        acc.val_abs_diffs.extend(ad)
        acc.val_signed_diffs.extend(sd)
        acc.val_structure_penalties.extend(sp)
        acc.batch_times_val.append(batch_time)

        row = {
            "timestamp": time.time(),
            "epoch": epoch,
            "batch_idx": batch_idx,
            "phase": "val",
            "total_loss": f"{total_loss:.6f}",
            "ce_loss": f"{ce_loss:.6f}",
            "value_loss": f"{value_loss:.6f}",
            "structure_loss": f"{structure_loss:.6f}",
            "parse_rate": f"{parse_rate:.4f}",
            "n_correct": nc,
            "n_wrong": nw,
            "n_unparseable": nu,
            "mean_abs_diff": f"{self._safe_mean(ad):.4f}" if ad else "",
            "median_abs_diff": f"{self._safe_median(ad):.4f}" if ad else "",
            "min_abs_diff": f"{self._safe_min(ad):.4f}" if ad else "",
            "max_abs_diff": f"{self._safe_max(ad):.4f}" if ad else "",
            "mean_signed_diff": f"{self._safe_mean(sd):.4f}" if sd else "",
            "std_abs_diff": f"{self._safe_std(ad):.4f}" if ad else "",
            "mean_structure_penalty": f"{self._safe_mean(sp):.4f}" if sp else "",
            "batch_time_sec": f"{batch_time:.4f}",
            "grad_norm": "",
            "lr": f"{lr:.8f}",
        }
        self._batch_writer.writerow(row)
        self._batch_file.flush()

    def end_epoch(
        self,
        epoch: int,
        lr: float,
        epoch_time_sec: float,
    ):
        """
        Finalize the epoch: compute aggregated stats and write one row
        to epoch_log.csv. Resets the accumulator for the next epoch.
        """
        acc = self._acc
        wall = time.time() - self._start_wall

        # Best val loss tracking
        val_mean = self._safe_mean(acc.val_losses)
        is_best = val_mean < self._best_val_loss and acc.val_losses
        if is_best:
            self._best_val_loss = val_mean

        # Train accuracy
        train_total_preds = acc.train_correct + acc.train_wrong + acc.train_unparseable
        train_acc = (
            100.0 * acc.train_correct / train_total_preds
            if train_total_preds > 0 else 0.0
        )

        # Val accuracy
        val_total_preds = acc.val_correct + acc.val_wrong + acc.val_unparseable
        val_acc = (
            100.0 * acc.val_correct / val_total_preds
            if val_total_preds > 0 else 0.0
        )

        # Bucket analysis
        train_buckets = self._bucket_pcts(acc.train_abs_diffs)
        val_buckets = self._bucket_pcts(acc.val_abs_diffs)

        row = {
            "timestamp": time.time(),
            "epoch": epoch,
            # ── Train loss ──
            "train_loss_mean": f"{self._safe_mean(acc.train_losses):.6f}",
            "train_loss_std": f"{self._safe_std(acc.train_losses):.6f}",
            "train_loss_min": f"{self._safe_min(acc.train_losses):.6f}",
            "train_loss_max": f"{self._safe_max(acc.train_losses):.6f}",
            "train_ce_loss_mean": f"{self._safe_mean(acc.train_ce_losses):.6f}",
            "train_value_loss_mean": f"{self._safe_mean(acc.train_value_losses):.6f}",
            "train_structure_loss_mean": f"{self._safe_mean(acc.train_structure_losses):.6f}",
            "train_parse_rate_mean": f"{self._safe_mean(acc.train_parse_rates):.4f}",
            # ── Train predictions ──
            "train_correct": acc.train_correct,
            "train_wrong": acc.train_wrong,
            "train_unparseable": acc.train_unparseable,
            "train_accuracy_pct": f"{train_acc:.2f}",
            "train_mean_abs_diff": f"{self._safe_mean(acc.train_abs_diffs):.4f}",
            "train_median_abs_diff": f"{self._safe_median(acc.train_abs_diffs):.4f}",
            "train_min_abs_diff": f"{self._safe_min(acc.train_abs_diffs):.4f}",
            "train_max_abs_diff": f"{self._safe_max(acc.train_abs_diffs):.4f}",
            "train_std_abs_diff": f"{self._safe_std(acc.train_abs_diffs):.4f}",
            "train_mean_signed_diff": f"{self._safe_mean(acc.train_signed_diffs):.4f}",
            "train_mean_structure_penalty": f"{self._safe_mean(acc.train_structure_penalties):.4f}",
            "train_median_structure_penalty": f"{self._safe_median(acc.train_structure_penalties):.4f}",
            # ── Val loss ──
            "val_loss_mean": f"{self._safe_mean(acc.val_losses):.6f}",
            "val_loss_std": f"{self._safe_std(acc.val_losses):.6f}",
            "val_loss_min": f"{self._safe_min(acc.val_losses):.6f}",
            "val_loss_max": f"{self._safe_max(acc.val_losses):.6f}",
            "val_ce_loss_mean": f"{self._safe_mean(acc.val_ce_losses):.6f}",
            "val_value_loss_mean": f"{self._safe_mean(acc.val_value_losses):.6f}",
            "val_structure_loss_mean": f"{self._safe_mean(acc.val_structure_losses):.6f}",
            "val_parse_rate_mean": f"{self._safe_mean(acc.val_parse_rates):.4f}",
            # ── Val predictions ──
            "val_correct": acc.val_correct,
            "val_wrong": acc.val_wrong,
            "val_unparseable": acc.val_unparseable,
            "val_accuracy_pct": f"{val_acc:.2f}",
            "val_mean_abs_diff": f"{self._safe_mean(acc.val_abs_diffs):.4f}",
            "val_median_abs_diff": f"{self._safe_median(acc.val_abs_diffs):.4f}",
            "val_min_abs_diff": f"{self._safe_min(acc.val_abs_diffs):.4f}",
            "val_max_abs_diff": f"{self._safe_max(acc.val_abs_diffs):.4f}",
            "val_std_abs_diff": f"{self._safe_std(acc.val_abs_diffs):.4f}",
            "val_mean_signed_diff": f"{self._safe_mean(acc.val_signed_diffs):.4f}",
            "val_mean_structure_penalty": f"{self._safe_mean(acc.val_structure_penalties):.4f}",
            "val_median_structure_penalty": f"{self._safe_median(acc.val_structure_penalties):.4f}",
            # ── Global ──
            "lr": f"{lr:.8f}",
            "epoch_time_sec": f"{epoch_time_sec:.2f}",
            "cumulative_train_samples": self._cumulative_samples,
            "cumulative_wall_time_sec": f"{wall:.2f}",
            "best_val_loss": f"{self._best_val_loss:.6f}",
            "is_best": int(is_best),
            "mean_grad_norm": f"{self._safe_mean(acc.grad_norms):.6f}",
            "max_grad_norm": f"{self._safe_max(acc.grad_norms):.6f}",
            # ── Buckets (train) ──
            "train_pct_off_by_0": f"{train_buckets['pct_off_by_0']:.2f}",
            "train_pct_off_by_le_1": f"{train_buckets['pct_off_by_le_1']:.2f}",
            "train_pct_off_by_le_5": f"{train_buckets['pct_off_by_le_5']:.2f}",
            "train_pct_off_by_le_10": f"{train_buckets['pct_off_by_le_10']:.2f}",
            "train_pct_off_by_le_100": f"{train_buckets['pct_off_by_le_100']:.2f}",
            # ── Buckets (val) ──
            "val_pct_off_by_0": f"{val_buckets['pct_off_by_0']:.2f}",
            "val_pct_off_by_le_1": f"{val_buckets['pct_off_by_le_1']:.2f}",
            "val_pct_off_by_le_5": f"{val_buckets['pct_off_by_le_5']:.2f}",
            "val_pct_off_by_le_10": f"{val_buckets['pct_off_by_le_10']:.2f}",
            "val_pct_off_by_le_100": f"{val_buckets['pct_off_by_le_100']:.2f}",
        }

        self._epoch_writer.writerow(row)
        self._epoch_file.flush()

        # Reset for next epoch
        self._acc.reset()

    def close(self):
        """Flush and close both CSV files."""
        self._batch_file.close()
        self._epoch_file.close()
