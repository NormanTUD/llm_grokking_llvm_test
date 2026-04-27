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

    while len(samples) < num_samples and attempts < num_samples * 10:
        attempts += 1

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

        # ── FIX: Encode the FULL text jointly, then truncate to prompt ──
        # This matches exactly how generate_single_sample() in train.py
        # constructs training samples. Encoding the prompt alone can
        # produce different BPE token boundaries, causing the model to
        # see a token sequence it was never trained on → immediate EOS.
        full_text = f"{ir_code}{expected_str}"
        full_ids = tokenizer.encode(full_text)
        prompt_only_ids = tokenizer.encode(ir_code)

        # Find the longest common prefix between prompt_only_ids and full_ids.
        # The prompt portion in the joint encoding is full_ids[:common_len].
        common_len = 0
        for a, b in zip(prompt_only_ids, full_ids):
            if a == b:
                common_len += 1
            else:
                break
        else:
            common_len = len(prompt_only_ids)

        # Build prompt_ids the same way training does: [bos] + joint_prefix
        prompt_ids = [bos_id] + full_ids[:common_len]

        # Greedy decode
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated_ids = []
        with torch.no_grad():
            for _ in range(max_gen_len):
                if input_tensor.shape[1] >= model.max_seq_len:
                    break
                output = model(input_ids=input_tensor)
                logits = output.logits[:, -1, :]
                next_id = logits.argmax(dim=-1).item()

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

        for special_tok in ["<eos>", "<pad>", "<bos>"]:
            predicted_str = predicted_str.replace(special_tok, "")
        predicted_str = predicted_str.strip()

        if not predicted_str:
            predicted_str = "(empty)"

        samples.append({
            "ir_code": ir_code.strip(),
            "params": ", ".join(str(p) for p in params),
            "expected": expected_str,
            "predicted": predicted_str,
            "correct": str(predicted_str == expected_str),
            "full_prompt": ir_code,
        })

    return samples
