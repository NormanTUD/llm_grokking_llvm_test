# generate_samples.py

"""
Helper to generate example sentences from the model for manual inspection.
Used by the run logger during training.
"""

import torch
from typing import List, Dict, Optional

from random_latex_gen import generate_random_function

def generate_example_samples(
    model,
    tokenizer,
    device: str,
    num_samples: int,
    max_params: int = 3,
    max_ops: int = 4,
    allowed_ops: Optional[List[str]] = None,
    param_range: tuple = (-20, 20),
    max_gen_len: int = 64,
) -> List[Dict[str, str]]:
    """
    Generate example input/output pairs and the model's predictions.

    Returns a list of dicts with keys:
        - ir_code: the LLVM IR source
        - params: comma-separated input params
        - expected: the correct integer result
        - predicted: the model's greedy-decoded output after <sep>
        - correct: whether predicted == expected
        - full_prompt: the full text fed to the model
    """
    import random

    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    model.eval()
    samples = []
    attempts = 0

    # Pre-compute special token IDs from the tokenizer
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
                num_params=num_p,
                params=params,
                allowed_ops=allowed_ops,
                num_operations=num_o,
                func_name="f",
            )
        except Exception:
            continue

        params_str = ",".join(str(p) for p in params)
        expected_str = str(expected_result)

        # Build the prompt: IR code + <sep> + params + <sep>
        # The model should predict the result after the second <sep>
        prompt_text = f"{ir_code}<sep>{params_str}<sep>"
        prompt_ids = [bos_id] + tokenizer.encode(prompt_text)

        # Greedy decode
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        generated_ids = []
        with torch.no_grad():
            for _ in range(max_gen_len):
                if input_tensor.shape[1] >= model.max_seq_len:
                    break

                output = model(input_ids=input_tensor)
                logits = output.logits[:, -1, :]  # last token
                next_id = logits.argmax(dim=-1).item()

                # Stop on EOS or PAD
                if next_id == eos_id:
                    break
                if next_id == pad_id:
                    break

                generated_ids.append(next_id)

                input_tensor = torch.cat(
                    [input_tensor, torch.tensor([[next_id]], device=device)],
                    dim=1,
                )

        # Decode all generated token IDs at once using the BPE tokenizer
        if generated_ids:
            predicted_str = tokenizer.decode(generated_ids).strip()
        else:
            predicted_str = ""

        # Clean up any special token remnants that might appear in decoded text
        for special_tok in ["<eos>", "<pad>", "<bos>", "<sep>"]:
            predicted_str = predicted_str.replace(special_tok, "")
        predicted_str = predicted_str.strip()

        samples.append({
            "ir_code": ir_code.strip(),
            "params": params_str,
            "expected": expected_str,
            "predicted": predicted_str,
            "correct": str(predicted_str == expected_str),
            "full_prompt": prompt_text,
        })

    return samples
