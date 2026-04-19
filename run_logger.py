# run_logger.py

"""
Logging utility for training runs.
Writes all training metadata, configs, loss tables, and example sentences
into runs/{run_id}/ with auto-incrementing run IDs.

Only writes changed data (compares before writing).
"""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RunLogger:
    """
    Logs training data into runs/{run_id}/.
    Auto-increments run_id if the folder already exists.
    Only writes files when content has changed.
    """

    def __init__(self, base_dir: str = "runs", run_id: Optional[int] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if run_id is not None:
            self.run_id = run_id
        else:
            self.run_id = self._next_run_id()

        self.run_dir = self.base_dir / str(self.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.samples_dir = self.run_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        # Cache of file hashes to avoid rewriting unchanged data
        self._file_hashes: Dict[str, str] = {}

        # Accumulated data
        self._epoch_losses: List[Dict[str, Any]] = []
        self._batch_losses_train: List[Dict[str, Any]] = []
        self._batch_losses_val: List[Dict[str, Any]] = []
        self._lr_history: List[Dict[str, Any]] = []

        # Write a timestamp file
        self._write_if_changed(
            self.run_dir / "started_at.txt",
            time.strftime("%Y-%m-%d %H:%M:%S %Z")
        )

    def get_base_dir(self):
        return self.base_dir

    def _next_run_id(self) -> int:
        """Find the next available run ID by scanning existing folders."""
        existing = []
        if self.base_dir.exists():
            for entry in self.base_dir.iterdir():
                if entry.is_dir():
                    try:
                        existing.append(int(entry.name))
                    except ValueError:
                        continue
        return max(existing, default=-1) + 1

    @staticmethod
    def _hash_content(content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _write_if_changed(self, path: Path, content: str) -> bool:
        """Write file only if content has changed. Returns True if written."""
        path = Path(path)
        new_hash = self._hash_content(content)
        key = str(path)

        # Check in-memory cache first
        if key in self._file_hashes and self._file_hashes[key] == new_hash:
            return False

        # Check existing file on disk
        if path.exists():
            existing_content = path.read_text(encoding="utf-8")
            if self._hash_content(existing_content) == new_hash:
                self._file_hashes[key] = new_hash
                return False

        path.write_text(content, encoding="utf-8")
        self._file_hashes[key] = new_hash
        return True

    def _write_csv_if_changed(
        self, path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]
    ) -> bool:
        """Write a CSV file only if content has changed."""
        import io
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return self._write_if_changed(path, buf.getvalue())

    # ── Config / Metadata ───────────────────────────────────────────────

    def log_config(self, args: argparse.Namespace):
        """Save the full argparse config as JSON and as a readable table CSV."""
        args_dict = vars(args)

        # JSON
        self._write_if_changed(
            self.run_dir / "config.json",
            json.dumps(args_dict, indent=2, default=str)
        )

        # CSV table
        rows = [{"parameter": k, "value": str(v)} for k, v in sorted(args_dict.items())]
        self._write_csv_if_changed(
            self.run_dir / "config.csv",
            rows,
            fieldnames=["parameter", "value"]
        )

    def log_model_summary(
        self,
        model_name: str,
        param_count: int,
        config_dict: Dict[str, Any],
        vocab_size: int,
    ):
        """Save model architecture summary."""
        summary = {
            "model_name": model_name,
            "total_parameters": param_count,
            "vocab_size": vocab_size,
            **config_dict,
        }
        self._write_if_changed(
            self.run_dir / "model_summary.json",
            json.dumps(summary, indent=2, default=str)
        )

    # ── Loss Logging ────────────────────────────────────────────────────

    def log_batch_loss_train(
        self, epoch: int, batch_idx: int, loss: float, ema_loss: float
    ):
        """Append a single training batch loss entry."""
        self._batch_losses_train.append({
            "epoch": epoch,
            "batch": batch_idx,
            "loss": round(loss, 6),
            "ema_loss": round(ema_loss, 6),
        })

    def log_batch_loss_val(self, epoch: int, batch_idx: int, loss: float):
        """Append a single validation batch loss entry."""
        self._batch_losses_val.append({
            "epoch": epoch,
            "batch": batch_idx,
            "loss": round(loss, 6),
        })

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        elapsed_secs: float,
        total_samples: int,
        is_best: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Log one epoch's summary."""
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "elapsed_secs": round(elapsed_secs, 2),
            "total_samples": total_samples,
            "is_best": is_best,
        }
        if extra:
            row.update(extra)
        self._epoch_losses.append(row)

        self._lr_history.append({
            "epoch": epoch,
            "lr": lr,
        })

    def flush_losses(self):
        """Write all accumulated loss CSVs to disk (only if changed)."""
        if self._epoch_losses:
            fieldnames = list(self._epoch_losses[0].keys())
            self._write_csv_if_changed(
                self.run_dir / "epoch_losses.csv",
                self._epoch_losses,
                fieldnames=fieldnames,
            )

        if self._batch_losses_train:
            self._write_csv_if_changed(
                self.run_dir / "batch_losses_train.csv",
                self._batch_losses_train,
                fieldnames=["epoch", "batch", "loss", "ema_loss"],
            )

        if self._batch_losses_val:
            self._write_csv_if_changed(
                self.run_dir / "batch_losses_val.csv",
                self._batch_losses_val,
                fieldnames=["epoch", "batch", "loss"],
            )

        if self._lr_history:
            self._write_csv_if_changed(
                self.run_dir / "lr_history.csv",
                self._lr_history,
                fieldnames=["epoch", "lr"],
            )

    # ── Example Sentences ───────────────────────────────────────────────

    def log_samples(
        self,
        epoch: int,
        samples: List[Dict[str, str]],
        n_samples: Optional[int] = None,
    ):
        """
        Write example samples for an epoch.

        Each sample dict should have keys like:
            - "input_ir": the LLVM IR code
            - "params": the parameter string
            - "expected": the expected result
            - "predicted": the model's predicted result (if available)
            - "full_text": the full decoded sequence

        Args:
            epoch: current epoch number
            samples: list of sample dicts
            n_samples: if set, only write the first n_samples. None = all.
        """
        if n_samples is not None:
            samples = samples[:n_samples]

        if not samples:
            return

        # Write as CSV
        fieldnames = list(samples[0].keys())
        self._write_csv_if_changed(
            self.samples_dir / f"epoch_{epoch:04d}.csv",
            samples,
            fieldnames=fieldnames,
        )

        # Also write a human-readable text version
        lines = []
        lines.append(f"{'='*80}")
        lines.append(f"EPOCH {epoch} — {len(samples)} sample(s)")
        lines.append(f"{'='*80}\n")

        for i, s in enumerate(samples):
            lines.append(f"--- Sample {i+1} ---")
            for k, v in s.items():
                # For multi-line values (like IR code), indent
                if "\n" in str(v):
                    lines.append(f"  {k}:")
                    for line in str(v).split("\n"):
                        lines.append(f"    {line}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        self._write_if_changed(
            self.samples_dir / f"epoch_{epoch:04d}.txt",
            "\n".join(lines)
        )

    # ── Final Summary ───────────────────────────────────────────────────

    def log_final_summary(
        self,
        train_losses: List[float],
        val_losses: List[float],
        best_val_loss: float,
        total_samples: int,
        total_epochs: int,
        total_time: str,
        param_count: int,
        save_path: Optional[str] = None,
    ):
        """Write the final training summary."""
        summary = {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "total_samples": total_samples,
            "total_epochs": total_epochs,
            "total_time": total_time,
            "model_parameters": param_count,
            "save_path": save_path,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        self._write_if_changed(
            self.run_dir / "final_summary.json",
            json.dumps(summary, indent=2)
        )

        # Also flush all pending CSVs
        self.flush_losses()

        # Write completion marker
        self._write_if_changed(
            self.run_dir / "completed_at.txt",
            time.strftime("%Y-%m-%d %H:%M:%S %Z")
        )

    @property
    def path(self) -> str:
        return str(self.run_dir)
