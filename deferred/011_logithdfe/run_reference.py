"""Run the reference on one workload in its own process (used by bench.py, which enforces a time limit).

usage: python run_reference.py <workload id> <out.json>
"""
import json
import sys

import reference
from workloads import WORKLOADS, generate

w = [x for x in WORKLOADS if x["id"] == int(sys.argv[1])][0]
r = reference.run(**generate(w))
json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()}, open(sys.argv[2], "w"))
