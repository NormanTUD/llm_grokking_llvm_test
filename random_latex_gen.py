# random_latex_gen.py

"""
Generates random math functions as LaTeX expressions,
evaluates them, and returns (latex_code, result).

Drop-in replacement for random_llvm_gen.py — same interface.
"""

import random
from typing import List, Optional, Tuple

# ── Supported operations ────────────────────────────────────────────────────

SUPPORTED_OPS = {
    "add":  ("Integer addition",      lambda a, b: a + b,  "{} + {}"),
    "sub":  ("Integer subtraction",   lambda a, b: a - b,  "{} - {}"),
    "mul":  ("Integer multiplication",lambda a, b: a * b,  "{} \\cdot {}"),
    "sdiv": ("Signed division",       lambda a, b: a // b if b != 0 else a, "\\frac{{{}}}{{{}}}"),
    "srem": ("Signed remainder",      lambda a, b: a % b if b != 0 else 0,  "{} \\bmod {}"),
    "shl":  ("Shift left",           lambda a, b: a << min(b % 64, 63),     "{} \\ll {}"),
    "ashr": ("Arith shift right",    lambda a, b: a >> min(b % 64, 63),     "{} \\gg {}"),
    "and":  ("Bitwise AND",          lambda a, b: a & b,  "{} \\land {}"),
    "or":   ("Bitwise OR",           lambda a, b: a | b,  "{} \\lor {}"),
    "xor":  ("Bitwise XOR",          lambda a, b: a ^ b,  "{} \\oplus {}"),
}


def list_supported_ops() -> dict:
    """Return a dict of supported operation names and their descriptions."""
    return {k: v[0] for k, v in SUPPORTED_OPS.items()}


def _validate_ops(allowed_ops: List[str]) -> None:
    for op in allowed_ops:
        if op not in SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported operation: '{op}'. "
                f"Supported: {list(SUPPORTED_OPS.keys())}"
            )


# ── Variable name generation ────────────────────────────────────────────────

def _param_names(n: int) -> List[str]:
    """Generate parameter names: x, y, z, w, a, b, c, ..."""
    names = list("xyzwabcdefghijklmnopqrstuv")
    if n <= len(names):
        return names[:n]
    return [f"x_{{{i}}}" for i in range(n)]


# ── Core generation function ────────────────────────────────────────────────

def generate_random_function(
    num_params: int,
    params: List[int],
    allowed_ops: Optional[List[str]] = None,
    num_operations: int = 5,
    seed: Optional[int] = None,
    func_name: str = "f",
) -> Tuple[str, int]:
    """
    Generate a random math function as a LaTeX expression, evaluate it.

    Parameters
    ----------
    num_params : int
        Number of parameters (>= 2).
    params : list of int
        Concrete parameter values. len(params) must equal num_params.
    allowed_ops : list of str, optional
        Which operations to use. Defaults to ["add", "sub"].
    num_operations : int
        How many binary operations to chain (>= 1).
    seed : int, optional
        Random seed for reproducibility.
    func_name : str
        Name of the function (used in LaTeX output).

    Returns
    -------
    (latex_code, result) : (str, int)
        latex_code – the LaTeX expression as a string
        result     – the integer result of evaluating with `params`
    """
    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    _validate_ops(allowed_ops)

    if num_params < 2:
        raise ValueError("num_params must be >= 2")
    if len(params) != num_params:
        raise ValueError(f"Expected {num_params} params, got {len(params)}")
    if num_operations < 1:
        raise ValueError("num_operations must be >= 1")

    if seed is not None:
        random.seed(seed)

    # ── Parameter names ─────────────────────────────────────────────────
    pnames = _param_names(num_params)

    # ── Build expression DAG (same topology as LLVM version) ────────────
    # Each entry in value_pool is (latex_expr_str, numeric_value)
    value_pool: List[Tuple[str, int]] = [
        (pnames[i], params[i]) for i in range(num_params)
    ]

    for i in range(num_operations):
        op_name = random.choice(allowed_ops)
        _, op_func, op_template = SUPPORTED_OPS[op_name]

        lhs_idx = random.randrange(len(value_pool))
        rhs_idx = random.randrange(len(value_pool))

        lhs_expr, lhs_val = value_pool[lhs_idx]
        rhs_expr, rhs_val = value_pool[rhs_idx]

        # Safety guards (matching LLVM version)
        rhs_eval = rhs_val
        if op_name in ("shl", "ashr", "lshr"):
            rhs_eval = rhs_val % 64
        if op_name in ("sdiv", "srem"):
            if rhs_val == 0:
                rhs_eval = 1

        # Compute numeric result
        try:
            result_val = op_func(lhs_val, rhs_eval)
            # Clamp to 64-bit signed range (matching LLVM i64)
            result_val = _clamp_i64(result_val)
        except (OverflowError, ZeroDivisionError):
            result_val = 0

        # Build LaTeX expression
        # Add parentheses for clarity when nesting
        lhs_tex = _maybe_paren(lhs_expr, op_name, "lhs")
        rhs_tex = _maybe_paren(rhs_expr, op_name, "rhs")
        result_expr = op_template.format(lhs_tex, rhs_tex)

        value_pool.append((result_expr, result_val))

    # The final value is the function's return value
    final_expr, final_val = value_pool[-1]

    # ── Build compact LaTeX output ──────────────────────────────────────
    param_list = ",".join(pnames)
    latex_code = f"{func_name}({param_list})={final_expr}"

    return latex_code, int(final_val)


def _clamp_i64(val: int) -> int:
    """Clamp to signed 64-bit integer range (matching LLVM i64 semantics)."""
    MOD = 1 << 64
    val = val % MOD
    if val >= (1 << 63):
        val -= MOD
    return val


def _maybe_paren(expr: str, op_name: str, side: str) -> str:
    """
    Add parentheses around sub-expressions when needed for clarity.
    
    Rules:
    - Single variables (x, y, z) never need parens
    - For mul/div, wrap add/sub sub-expressions
    - For frac{}{}, no parens needed (LaTeX handles it)
    """
    # Single variable or number — no parens needed
    if len(expr) <= 3 and not any(c in expr for c in "+-\\"):
        return expr

    # frac and bmod handle grouping via LaTeX braces
    if op_name in ("sdiv", "srem"):
        return expr

    # For multiplication, parenthesize additions/subtractions
    if op_name == "mul" and any(op in expr for op in [" + ", " - "]):
        return f"({expr})"

    return expr


# ── Convenience: same API as random_llvm_gen ────────────────────────────────

def generate_with_native_asm(
    num_params: int,
    params: List[int],
    allowed_ops: Optional[List[str]] = None,
    num_operations: int = 5,
    seed: Optional[int] = None,
    func_name: str = "f",
) -> Tuple[str, str, int]:
    """
    Compatibility shim. Returns (latex_code, "", result).
    The second element (native_asm) is empty since we don't use LLVM.
    """
    latex_code, result = generate_random_function(
        num_params=num_params,
        params=params,
        allowed_ops=allowed_ops,
        num_operations=num_operations,
        seed=seed,
        func_name=func_name,
    )
    return latex_code, "", result


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Supported operations: {list(list_supported_ops().keys())}")
    print()

    # Same test case as random_llvm_gen.py
    latex_code, result = generate_random_function(
        num_params=3,
        params=[10, 20, 30],
        allowed_ops=["add", "mul", "sub"],
        num_operations=4,
        seed=42,
    )

    print("=" * 60)
    print("LaTeX expression:")
    print("=" * 60)
    print(latex_code)
    print("=" * 60)
    print(f"Result with params (10, 20, 30): {result}")
    print()

    # Show token count comparison
    print("Token count comparison:")
    print(f"  LaTeX: {len(latex_code)} chars")

    # Generate a few more examples
    print("\nMore examples:")
    for seed in range(5):
        code, res = generate_random_function(
            num_params=2,
            params=[5, 3],
            allowed_ops=["add", "sub", "mul"],
            num_operations=3,
            seed=seed,
        )
        print(f"  {code}  →  {res}")
