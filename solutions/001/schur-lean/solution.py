"""Task 001 candidate "schur-lean": entry point. The computation is in schur_lean.py, imported under a
fixed module name so that its Numba kernels load from the cache built ahead of the evaluation (see
build.md; cached code refers to its module by name). The cache is read from a private temporary
copy, so that a compilation for an argument type it lacks is written there and never into the
candidate's directory (the evaluator's snapshot, which must not change)."""
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

_NAME = "socialscibench_001_schur_lean"
if _NAME not in sys.modules:
    _kernels = Path(__file__).with_name("schur_lean.py")
    if not os.environ.get("SOCIALSCIBENCH_BUILD_CACHE"):          # build_cache.py writes the cache in place
        import numba
        from numba.core.caching import UserProvidedCacheLocator
        _root = tempfile.mkdtemp(prefix="ssb-numba-cache-")
        _dest = os.path.join(_root, UserProvidedCacheLocator.get_suitable_cache_subpath(str(_kernels)))
        if (_kernels.parent / "__pycache__").is_dir():
            shutil.copytree(_kernels.parent / "__pycache__", _dest, copy_function=shutil.copy2)
        numba.config.CACHE_DIR = _root
    _spec = importlib.util.spec_from_file_location(_NAME, _kernels)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)
run = sys.modules[_NAME].run
