# random_infix_gen.py — Simplest possible representation

import random
from typing import List, Optional, Tuple

SUPPORTED_OPS = {
    "add":  (lambda a, b: a + b,  "+",  1),
    "sub":  (lambda a, b: a - b,  "-",  1),
    "mul":  (lambda a, b: a * b,  "*",  2),
    "sdiv": (lambda a, b: a // b if b != 0 else a, "/", 2),
    "srem": (lambda a, b: a % b if b != 0 else 0,  "%", 2),
    "shl":  (lambda a, b: a << min(b % 64, 63),    "<<", 3),
    "ashr": (lambda a, b: a >> min(b % 64, 63),    ">>", 3),
    "and":  (lambda a, b: a & b,  "&",  0),
    "or":   (lambda a, b: a | b,  "|",  0),
    "xor":  (lambda a, b: a ^ b,  "^",  0),
}

def list_supported_ops():
    return {k: k for k in SUPPORTED_OPS}

def _param_names(n):
    names = list("xyzwabcdefghijklmnopqrstuv")
    return names[:n] if n <= len(names) else [f"x{i}" for i in range(n)]

def _clamp_i64(val):
    MOD = 1 << 64
    val = val % MOD
    if val >= (1 << 63):
        val -= MOD
    return val

def generate_random_function(
    num_params: int,
    params: List[int],
    allowed_ops: Optional[List[str]] = None,
    num_operations: int = 5,
    seed: Optional[int] = None,
    func_name: str = "f",
) -> Tuple[str, int]:

    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    if num_params < 2:
        raise ValueError("num_params must be >= 2")
    if len(params) != num_params:
        raise ValueError(f"Expected {num_params} params, got {len(params)}")

    if seed is not None:
        random.seed(seed)

    pnames = _param_names(num_params)

    # (expr_string, numeric_value, precedence_of_outer_op)
    value_pool: List[Tuple[str, int, int]] = [
        (pnames[i], params[i], 99) for i in range(num_params)
    ]

    for _ in range(num_operations):
        op_name = random.choice(allowed_ops)
        op_func, op_sym, op_prec = SUPPORTED_OPS[op_name]

        li = random.randrange(len(value_pool))
        ri = random.randrange(len(value_pool))

        le, lv, lp = value_pool[li]
        re, rv, rp = value_pool[ri]

        # Safety
        rv_safe = rv
        if op_name in ("shl", "ashr"):
            rv_safe = rv % 64
        if op_name in ("sdiv", "srem") and rv == 0:
            rv_safe = 1

        try:
            result_val = _clamp_i64(op_func(lv, rv_safe))
        except:
            result_val = 0

        # Parenthesization: only when child has lower precedence
        l_str = f"({le})" if lp < op_prec else le
        r_str = f"({re})" if rp <= op_prec else re  # <= for right-assoc safety

        expr = f"{l_str}{op_sym}{r_str}"
        value_pool.append((expr, result_val, op_prec))

    final_expr, final_val, _ = value_pool[-1]
    param_list = ",".join(pnames)
    code = f"{func_name}({param_list})={final_expr}"

    return code, int(final_val)


if __name__ == "__main__":
    for s in range(8):
        code, res = generate_random_function(
            num_params=3, params=[10, 5, 3],
            allowed_ops=["add", "sub", "mul", "sdiv"],
            num_operations=4, seed=s,
        )
        print(f"  {code}  =  {res}")
