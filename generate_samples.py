# generate_samples.py

"""
Helper to generate example sentences from the model for manual inspection.
Used by the run logger during training.
"""

import torch
from typing import List, Dict, Optional

from random_infix_gen import generate_random_function

def generate_example_samples(
    model, tokenizer, device, num_samples,
    max_params=3, max_ops=4, allowed_ops=None,
    param_range=(-20, 20), max_gen_len=64,
    task="infix",
    max_turnstiles=10,
):
    import random

    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    model.eval()
    samples = []
    attempts = 0

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id

    # Detect if the model is a classifier (has cls_head outputting (B, num_classes))
    is_classifier = hasattr(model, 'cls_head')

    while len(samples) < num_samples and attempts < num_samples * 10:
        attempts += 1

        # ── Generate the prompt + expected answer ───────────────────
        if task == "turnstile":
            from turnstile_gen import generate_turnstile_function
            try:
                ir_code, expected_result = generate_turnstile_function(
                    max_turnstiles=max_turnstiles,
                )
            except Exception:
                continue
            expected_str = str(expected_result)
            params_str = ""
        else:
            num_p = random.randint(2, max(2, max_params))
            num_o = random.randint(1, max(1, max_ops))
            params = [random.randint(*param_range) for _ in range(num_p)]

            try:
                ir_code, expected_result = generate_random_function(
                    num_params=num_p, params=params,
                    allowed_ops=allowed_ops, num_operations=num_o,
                    func_name="f",
                )
            except Exception:
                continue
            expected_str = str(expected_result)
            params_str = ", ".join(str(p) for p in params)

        # ── Encode prompt using joint BPE (same as training) ────────
        full_text = f"{ir_code}{expected_str}"
        full_ids = tokenizer.encode(full_text)
        prompt_only_ids = tokenizer.encode(ir_code)

        common_len = 0
        for a, b in zip(prompt_only_ids, full_ids):
            if a == b:
                common_len += 1
            else:
                break
        else:
            common_len = len(prompt_only_ids)

        prompt_ids = [bos_id] + full_ids[:common_len]

        # ── Classification mode: use for turnstile OR classifier model ──
        if task == "turnstile" or is_classifier:
            # Feed prompt, get 2-class logits, argmax
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                output = model(input_ids=input_tensor)
                if isinstance(output, tuple):
                    _, logits = output
                else:
                    logits = output.logits
                predicted_class = logits.argmax(dim=-1).item()
            predicted_str = str(predicted_class)
        else:
            # Greedy autoregressive decode (only for true language models)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated_ids = []
            with torch.no_grad():
                for _ in range(max_gen_len):
                    if input_tensor.shape[1] >= model.max_seq_len:
                        break
                    output = model(input_ids=input_tensor, output_hidden_states=False)
                    # Handle both tuple and named output
                    if isinstance(output, tuple):
                        _, step_logits = output
                    else:
                        step_logits = output.logits

                    # Safety check: if logits are 2D (B, C) instead of 3D (B, T, V),
                    # this is a classifier model — use argmax directly and stop
                    if step_logits.dim() == 2:
                        next_id = step_logits.argmax(dim=-1).item()
                        if next_id != eos_id and next_id != pad_id:
                            generated_ids.append(next_id)
                        break

                    # For autoregressive: logits are (B, T, V)
                    next_logits = step_logits[:, -1, :]
                    next_id = next_logits.argmax(dim=-1).item()

                    if next_id == eos_id or next_id == pad_id:
                        break

                    generated_ids.append(next_id)
                    input_tensor = torch.cat(
                        [input_tensor, torch.tensor([[next_id]], device=device)],
                        dim=1,
                    )

            if generated_ids:
                predicted_str = tokenizer.decode(generated_ids).strip()
            else:
                predicted_str = ""

        # ── Clean ───────────────────────────────────────────────────
        for special_tok in ["<eos>", "<pad>", "<bos>"]:
            predicted_str = predicted_str.replace(special_tok, "")
        predicted_str = predicted_str.strip()

        if not predicted_str:
            predicted_str = "(empty)"

        samples.append({
            "ir_code": ir_code.strip(),
            "params": params_str,
            "expected": expected_str,
            "predicted": predicted_str,
            "correct": str(predicted_str == expected_str),
            "full_prompt": ir_code,
        })

    return samples
