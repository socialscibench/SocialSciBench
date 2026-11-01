"""Command line: turbo-socialsci {hdfe,did,reclink,warmup} ...  (exit 3 = not applicable)."""
import argparse
import json
import sys


def _out(frames, out):
    """Write one CSV per table (out may be a .csv path for a single table, or a directory)."""
    from pathlib import Path
    if out is None:
        for name, f in frames.items():
            print(f"# {name}")
            print(f.to_csv(index=True))
        return
    p = Path(out)
    if len(frames) == 1 and p.suffix == ".csv":
        next(iter(frames.values())).to_csv(p, index=True)
        return
    p.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_csv(p / f"{name}.csv", index=True)


def main(argv=None):
    from . import NotApplicable
    ap = argparse.ArgumentParser(prog="turbo-socialsci")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hdfe", help="reghdfe-style OLS with absorbed fixed effects, clustered SEs")
    h.add_argument("--data", required=True)
    h.add_argument("--y", required=True)
    h.add_argument("--x", nargs="+", required=True)
    h.add_argument("--absorb", nargs="+", required=True, help="fixed effects; a#b for interactions")
    h.add_argument("--cluster", required=True)
    d = sub.add_parser("did", help="csdid-style Callaway & Sant'Anna staggered DiD")
    d.add_argument("--data", required=True)
    for a in ("--y", "--unit", "--time", "--cohort"):
        d.add_argument(a, required=True)
    d.add_argument("--covariates", nargs="*", default=[])
    d.add_argument("--weights")
    d.add_argument("--control", default="never", choices=["never", "notyet"])
    d.add_argument("--method", default="dripw", choices=["dripw", "drimp", "reg", "stdipw", "ipw"])
    d.add_argument("--base-period", default="varying", choices=["varying", "universal"])
    d.add_argument("--trim", type=float, default=1.0)
    d.add_argument("--rules", default="csdid181", choices=["csdid181", "csdid201"])
    d.add_argument("--n-boot", type=int)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--alpha", type=float, default=0.05)
    d.add_argument("--event-window", type=int, nargs=2)
    d.add_argument("--boot-tables", nargs="+", choices=["gt", "event", "group", "calendar"])
    r = sub.add_parser("reclink", help="reclink2-style name and address record linkage")
    r.add_argument("--master", required=True)
    r.add_argument("--using", required=True)
    r.add_argument("--idmaster", required=True)
    r.add_argument("--idusing", required=True)
    r.add_argument("--vars", nargs="+", required=True)
    r.add_argument("--wmatch", type=float, nargs="+", required=True)
    r.add_argument("--wnomatch", type=float, nargs="+", required=True)
    r.add_argument("--required", nargs="*", default=[])
    r.add_argument("--minscore", type=float, default=0.6)
    r.add_argument("--minbigram", type=float, default=0.6)
    w = sub.add_parser("warmup", help="compile or load the fast kernels once")
    w.add_argument("which", nargs="*", default=["hdfe", "did", "reclink"])
    for p in (h, d, r):
        p.add_argument("--variant", default="auto" if p is not r else "speed", choices=["auto", "speed", "memory"])
        p.add_argument("--threads", type=int)
        p.add_argument("--check", action="store_true", help="only check that the request is covered")
        p.add_argument("--out", help="output CSV (one table) or directory (several tables)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "warmup":
            from . import warmup
            warmup(args.which)
            print("warm")
            return 0
        if args.cmd == "hdfe":
            from .hdfe import feols
            res = feols(args.data, args.y, args.x, args.absorb, args.cluster, variant=args.variant,
                        threads=args.threads, check_only=args.check)
            if args.check:
                print(json.dumps(res))
                return 0
            print(res.summary(), file=sys.stderr)
            _out({"coef": res.coef}, args.out)
            return 0
        if args.cmd == "did":
            from .did import att_gt
            res = att_gt(args.data, args.y, args.unit, args.time, args.cohort, covariates=args.covariates,
                         weights=args.weights, control=args.control, method=args.method,
                         base_period=args.base_period, trim=args.trim, rules=args.rules, n_boot=args.n_boot,
                         seed=args.seed, alpha=args.alpha, event_window=args.event_window,
                         boot_tables=args.boot_tables,
                         variant=args.variant, threads=args.threads, check_only=args.check)
            if args.check:
                print(json.dumps(res))
                return 0
            print(json.dumps(dict(crit=res.crit, variant=res.variant)), file=sys.stderr)
            _out(dict(gt=res.gt, event=res.event, group=res.group, calendar=res.calendar,
                      averages=res.averages), args.out)
            return 0
        if args.cmd == "reclink":
            from .reclink import reclink
            res = reclink(args.master, args.using, idmaster=args.idmaster, idusing=args.idusing, vars=args.vars,
                          wmatch=args.wmatch, wnomatch=args.wnomatch, required=args.required,
                          minscore=args.minscore, minbigram=args.minbigram, variant=args.variant,
                          threads=args.threads, check_only=args.check)
            if args.check:
                print(json.dumps(res))
                return 0
            print(json.dumps(dict(pairs=len(res), n_perfect=res.attrs["n_perfect"])), file=sys.stderr)
            if args.out:
                res.to_csv(args.out, index=False)
            else:
                print(res.to_csv(index=False))
            return 0
    except NotApplicable as exc:
        print(f"turbo-socialsci: not applicable: {exc}", file=sys.stderr)
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
