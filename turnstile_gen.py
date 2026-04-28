# turnstile_gen.py — Generate ⊢-counting equality sentences

"""
Generates sentences like:
    ⊢x = x → label 0  (true:  1 ⊢ on left, 0 on right... wait)

The task:
  - Both sides of '=' have some number of '⊢' prefixed to a variable (e.g. 'x')
  - The sentence evaluates to 1 (true) if the count of '⊢' is equal on both sides,
    and 0 (false) otherwise.

Examples:
    ⊢x = ⊢x          → 1   (1 == 1)
    ⊢⊢x = ⊢x         → 0   (2 != 1)
    ⊢⊢⊢x = ⊢⊢⊢x     → 1   (3 == 3)
    x = x              → 1   (0 == 0)
    ⊢⊢x = ⊢⊢⊢⊢x     → 0   (2 != 4)
"""

import random
from typing import Optional, Tuple


def generate_turnstile_sample(
    max_turnstiles: int = 10,
    seed: Optional[int] = None,
    var_name: str = "x",
    bias_equal: float = 0.5,
) -> Tuple[str, int]:
    """
    Generate a single turnstile-counting sample.

    Args:
        max_turnstiles: Maximum number of ⊢ on either side.
        seed:           Optional random seed for reproducibility.
        var_name:       Variable name to use (default 'x').
        bias_equal:     Probability of generating an equal (label=1) sample.
                        0.5 = balanced. Helps avoid class imbalance.

    Returns:
        (sentence, label)
        where sentence is like "⊢⊢x=⊢x " and label is 0 or 1.
    """
    if seed is not None:
        random.seed(seed)

    # Decide if this sample should be equal or not
    if random.random() < bias_equal:
        # Generate equal counts
        n = random.randint(0, max_turnstiles)
        left_count = n
        right_count = n
        label = 1
    else:
        # Generate unequal counts
        left_count = random.randint(0, max_turnstiles)
        right_count = random.randint(0, max_turnstiles)
        # Make sure they're actually different
        while right_count == left_count:
            right_count = random.randint(0, max_turnstiles)
        label = 0

    left_side = "⊢" * left_count + var_name
    right_side = "⊢" * right_count + var_name

    # Format: "⊢⊢x=⊢x " (no spaces around =, trailing space before answer)
    sentence = f"{left_side}={right_side} "

    return sentence, label


def generate_turnstile_function(
    max_turnstiles: int = 10,
    seed: Optional[int] = None,
    var_name: str = "x",
    bias_equal: float = 0.5,
    **kwargs,  # Accept and ignore extra kwargs for compatibility
) -> Tuple[str, int]:
    """
    Wrapper with the same signature pattern as generate_random_function.

    Returns:
        (prompt_str, result_int)
        where prompt_str ends with a space (ready for the answer to be appended)
        and result_int is 0 or 1.
    """
    return generate_turnstile_sample(
        max_turnstiles=max_turnstiles,
        seed=seed,
        var_name=var_name,
        bias_equal=bias_equal,
    )


if __name__ == "__main__":
    print("Sample turnstile sentences:")
    print("-" * 40)
    for i in range(20):
        sentence, label = generate_turnstile_sample(
            max_turnstiles=6, seed=i
        )
        print(f"  {sentence}{label}")

