"""One reference or candidate process. The controller performs all comparisons."""
import gc
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .common import entrypoint, read_json, task_hook, write_json
from .validation import normalize


def fingerprint(value):
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype.str)).encode())
            # NumPy exposes zero-column arrays (csdid 106) with zero strides;
            # memoryview.cast rejects them even though there are no bytes to hash.
            if v.size:
                h.update(memoryview(np.ascontiguousarray(v)).cast("B"))
        elif isinstance(v, dict):
            for key in sorted(v):
                h.update(key.encode())
                visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(json.dumps(v).encode())

    visit(value)
    return h.hexdigest()


def freeze(value):
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            freeze(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze(item)


def environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                threadpools=threadpool_info())


def execute(config):
    out = Path(config["output_dir"])
    result = dict(status="RUNTIME_ERROR", outputs=[], samples_s=[])
    try:
        task_dir = Path(config["task_dir"])
        manifest = read_json(task_dir / "task.json")
        generate = task_hook(task_dir, manifest, "workload_entry_point")
        adapt = task_hook(task_dir, manifest, "adapter")
        if config["case"]["kind"] == "real":
            from .datasets import load_inputs
            data = load_inputs(task_dir, manifest, config["case"], config["data_root"])
        else:
            data = adapt(task_dir, config["task"], config["case"], generate)
        freeze(data)
        result["input_sha256"] = fingerprint(data)
        result["dimensions"] = ({"n_regressors": data["X"].shape[1]} if config["task"] == "001" else
                                {"n_master": len(data["master"]["name"]), "n_using": len(data["using"]["name"])}
                                if config["task"] == "003" else {})
        t0 = time.perf_counter()
        run = entrypoint(Path(config["source_dir"]), config["entry_point"])
        sync = (entrypoint(Path(config["source_dir"]), config["synchronize"])
                if config.get("synchronize") else lambda: None)
        result["import_setup_s"] = time.perf_counter() - t0
        with threadpool_limits(limits=config["threads"]):
            phases = ["first_call"] + ["warmup"] * config["warmups"] + ["measured"] * config["repeats"]
            for index, phase in enumerate(phases):
                gc.collect()
                sync()
                t0 = time.perf_counter()
                output = run(**data)
                sync()
                elapsed = time.perf_counter() - t0
                if phase == "first_call":
                    result["first_call_s"] = elapsed
                if phase == "measured":
                    result["samples_s"].append(elapsed)
                try:
                    arrays = normalize(config["task"], output)
                    if any(v.dtype.hasobject for v in arrays.values()):
                        raise ValueError("Outputs must contain numeric arrays, not Python objects")
                    filename = f"output-{index}.npz"
                    np.savez(out / filename, **arrays)
                except (ValueError, TypeError) as error:
                    result.update(status="INVALID_OUTPUT", error=str(error))
                    break
                result["outputs"].append(dict(file=filename, phase=phase))
                del output, arrays
                if fingerprint(data) != result["input_sha256"]:
                    result.update(status="INPUT_MUTATION", error="Candidate changed its inputs")
                    break
                if config.get("stream_outputs"):
                    write_json(out / "output-ready.json", dict(index=index, file=filename, phase=phase,
                               dimensions=result["dimensions"], input_sha256=result["input_sha256"]))
                    ack = out / "output-ack.json"
                    while not ack.exists():
                        time.sleep(0.01)
                    if read_json(ack)["index"] != index:
                        raise ValueError("Output acknowledgement does not match repetition")
                    ack.unlink()
            else:
                result["status"] = "COMPLETED"
            result["environment"] = environment()
    except Exception as error:
        result.update(status="RUNTIME_ERROR", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        # Includes interpreter, generation, imports, outputs and serialization; excludes children.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        write_json(out / "result.json", result)


if __name__ == "__main__":
    config_path = Path(sys.argv[1]).resolve()
    config = read_json(config_path)
    for key in ("task_dir", "source_dir", "output_dir"):
        config[key] = str((config_path.parent / config[key]).resolve())
    execute(config)
