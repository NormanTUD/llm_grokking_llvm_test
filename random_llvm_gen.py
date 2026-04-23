# random_llvm_gen.py

"""
A Python module that generates random LLVM IR math functions,
returns the assembly, and executes them with given parameters.

Dependencies:
    pip install llvmlite

Usage:
    from random_llvm_gen import generate_random_function

    ir_code, result = generate_random_function(
        num_params=3,
        params=[10, 20, 30],
        allowed_ops=["add", "mul", "sub"],
        num_operations=5,
        seed=42,
    )
    print(ir_code)
    print(f"Result: {result}")
"""

import random
from ctypes import CFUNCTYPE, c_int64
from typing import List, Optional, Tuple

import llvmlite.ir as ir
import llvmlite.binding as llvm

# ── Initialize LLVM targets ────────────────────────────────────────────────
# Newer llvmlite removed the blanket initialize() but still requires
# explicit target / asm-printer init before you can JIT or emit asm.
_llvm_initialized = False

def _ensure_llvm_initialized():
    global _llvm_initialized
    if _llvm_initialized:
        return
    try:
        llvm.initialize()
    except RuntimeError:
        pass
    try:
        llvm.initialize_native_target()
    except RuntimeError:
        pass
    try:
        llvm.initialize_native_asmprinter()
    except RuntimeError:
        pass
    # If none of the above worked, try the all-targets approach
    try:
        llvm.initialize_all_targets()
    except (RuntimeError, AttributeError):
        pass
    try:
        llvm.initialize_all_asmprinters()
    except (RuntimeError, AttributeError):
        pass
    _llvm_initialized = True


# ── Supported operations ────────────────────────────────────────────────────

SUPPORTED_OPS = {
    "add":  "Integer addition (a + b)",
    "sub":  "Integer subtraction (a - b)",
    "mul":  "Integer multiplication (a * b)",
    "shl":  "Shift left (a << b)",
    "ashr": "Arithmetic shift right (a >> b)",
    "lshr": "Logical shift right",
    "and":  "Bitwise AND (a & b)",
    "or":   "Bitwise OR (a | b)",
    "xor":  "Bitwise XOR (a ^ b)",
    "sdiv": "Signed integer division (a / b) — divisor clamped to avoid div-by-zero",
    "srem": "Signed integer remainder (a % b) — divisor clamped to avoid div-by-zero",
}


def list_supported_ops() -> dict:
    """Return a dict of supported operation names and their descriptions."""
    return dict(SUPPORTED_OPS)


# ── Validation ──────────────────────────────────────────────────────────────

def _validate_ops(allowed_ops: List[str]) -> None:
    for op in allowed_ops:
        if op not in SUPPORTED_OPS:
            raise ValueError(
                f"Unsupported operation: '{op}'. "
                f"Supported operations are: {list(SUPPORTED_OPS.keys())}"
            )


# ── IR builder helpers ──────────────────────────────────────────────────────

def _emit_op(
    builder: ir.IRBuilder,
    op_name: str,
    lhs: ir.Value,
    rhs: ir.Value,
    name: str,
    i64: ir.IntType,
    op_index: int,
) -> ir.Value:
    """Emit a single LLVM IR binary instruction with safety guards."""

    # For shifts, clamp RHS to [0, 63] to avoid undefined behaviour
    if op_name in ("shl", "ashr", "lshr"):
        rhs = builder.urem(rhs, ir.Constant(i64, 64), name=f"shclamp{op_index}")

    # For division/remainder, guard against division by zero
    if op_name in ("sdiv", "srem"):
        is_zero = builder.icmp_unsigned("==", rhs, ir.Constant(i64, 0), name=f"iszero{op_index}")
        rhs = builder.select(is_zero, ir.Constant(i64, 1), rhs, name=f"safediv{op_index}")

    dispatch = {
        "add":  builder.add,
        "sub":  builder.sub,
        "mul":  builder.mul,
        "shl":  builder.shl,
        "ashr": builder.ashr,
        "lshr": builder.lshr,
        "and":  builder.and_,
        "or":   builder.or_,
        "xor":  builder.xor,
        "sdiv": builder.sdiv,
        "srem": builder.srem,
    }
    return dispatch[op_name](lhs, rhs, name=name)


# ── Core generation function ────────────────────────────────────────────────

def generate_random_function(
    num_params: int,
    params: List[int],
    allowed_ops: Optional[List[str]] = None,
    num_operations: int = 5,
    seed: Optional[int] = None,
    func_name: str = "random_func",
) -> Tuple[str, int]:
    """
    Generate a random LLVM IR function, JIT-compile it, and execute it.

    Parameters
    ----------
    num_params : int
        Number of parameters the generated function accepts (>= 2).
    params : list of int
        Concrete parameter values to feed into the function.
        len(params) must equal num_params.
    allowed_ops : list of str, optional
        Which operations to use.  Defaults to ["add", "sub"].
        See list_supported_ops() for all valid names.
    num_operations : int
        How many binary operations to chain together (>= 1).
    seed : int, optional
        Random seed for reproducibility.
    func_name : str
        Name of the generated LLVM function.

    Returns
    -------
    (ir_code, result) : (str, int)
        ir_code  – the LLVM IR assembly as a string
        result   – the integer result of calling the function with `params`

    Raises
    ------
    ValueError
        If an unsupported operation is requested, or if arguments are invalid.
    """

    # ── Defaults & validation ───────────────────────────────────────────
    if allowed_ops is None:
        allowed_ops = ["add", "sub"]

    _validate_ops(allowed_ops)

    if num_params < 2:
        raise ValueError("num_params must be >= 2")
    if len(params) != num_params:
        raise ValueError(
            f"Expected {num_params} parameter values, got {len(params)}"
        )
    if num_operations < 1:
        raise ValueError("num_operations must be >= 1")

    if seed is not None:
        random.seed(seed)

    # ── Build the LLVM IR module ────────────────────────────────────────
    i64 = ir.IntType(64)
    module = ir.Module(name="random_module")
    func_type = ir.FunctionType(i64, [i64] * num_params)
    function = ir.Function(module, func_type, name=func_name)

    for idx, arg in enumerate(function.args):
        arg.name = f"p{idx}"

    block = function.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # ── Generate random expression DAG ──────────────────────────────────
    value_pool: List[ir.Value] = list(function.args)

    for i in range(num_operations):
        op_name = random.choice(allowed_ops)
        lhs = random.choice(value_pool)
        rhs = random.choice(value_pool)

        result_val = _emit_op(builder, op_name, lhs, rhs, name=f"t{i}", i64=i64, op_index=i)
        value_pool.append(result_val)

    builder.ret(value_pool[-1])

    # ── Get the IR string ───────────────────────────────────────────────
    ir_code = str(module)

    # ── JIT compile & execute ───────────────────────────────────────────
    _ensure_llvm_initialized()

    llvm_module = llvm.parse_assembly(ir_code)
    llvm_module.verify()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    engine = llvm.create_mcjit_compiler(llvm_module, target_machine)

    func_ptr = engine.get_function_address(func_name)

    c_func_type = CFUNCTYPE(c_int64, *([c_int64] * num_params))
    callable_func = c_func_type(func_ptr)

    result = callable_func(*params)

    return ir_code, result


# ── Convenience: get native assembly too ────────────────────────────────────

def generate_with_native_asm(
    num_params: int,
    params: List[int],
    allowed_ops: Optional[List[str]] = None,
    num_operations: int = 5,
    seed: Optional[int] = None,
    func_name: str = "random_func",
) -> Tuple[str, str, int]:
    """
    Same as generate_random_function but also returns native (x86/ARM/…)
    assembly as a third element.

    Returns
    -------
    (ir_code, native_asm, result)
    """
    ir_code, result = generate_random_function(
        num_params=num_params,
        params=params,
        allowed_ops=allowed_ops,
        num_operations=num_operations,
        seed=seed,
        func_name=func_name,
    )

    _ensure_llvm_initialized()

    llvm_module = llvm.parse_assembly(ir_code)
    llvm_module.verify()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    native_asm = target_machine.emit_assembly(llvm_module)

    return ir_code, native_asm, result


# ── Quick demo when run directly ────────────────────────────────────────────

if __name__ == "__main__":
    # Print llvmlite version for debugging
    print(f"llvmlite version: {llvm.llvm_version_info}")
    print(f"Supported operations: {list(list_supported_ops().keys())}")

    # Debug: check what initialization functions are available
    init_funcs = [attr for attr in dir(llvm) if "init" in attr.lower()]
    print(f"Available init functions: {init_funcs}")
    print()

    _ensure_llvm_initialized()

    ir_code, result = generate_random_function(
        num_params=3,
        params=[10, 20, 30],
        allowed_ops=["add", "mul", "sub"],
        num_operations=4,
        seed=42,
    )

    print("=" * 60)
    print("LLVM IR:")
    print("=" * 60)
    print(ir_code)
    print("=" * 60)
    print(f"Result with params (10, 20, 30): {result}")
