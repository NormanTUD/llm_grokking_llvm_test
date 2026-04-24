#!/usr/bin/env bash
set -euo pipefail

# Test that train.py is valid Python (syntax check only — no execution)
python3 -c "
import py_compile, sys
try:
    py_compile.compile('train.py', doraise=True)
    print('✓ train.py syntax OK')
except py_compile.PyCompileError as e:
    print(f'✗ Syntax error: {e}', file=sys.stderr)
    sys.exit(1)
"

exit 0
