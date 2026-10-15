"""Command-line entry point.

Examples::

    python -m self_organized_codes run --set noise=0.05 n=12 n_roles=4
    python -m self_organized_codes sweep noise 0,0.02,0.05,0.1 --seeds 30 --out results/noise.csv
"""

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from self_organized_codes.config import SimConfig
from self_organized_codes.simulation import Simulation

logger = logging.getLogger("self_organized_codes")

SUMMARY_COLUMNS = ("code_size", "d", "d_roles", "d_top", "roles_covered", "purity_mean", "word_error", "role_error")


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings into typed :class:`SimConfig` fields.

    :param items: overrides as given on the command line, e.g. ``["noise=0.05", "n=12"]``.
    :return: field names mapped to their parsed values.
    :raises SystemExit: on an unknown key or an unparsable value.
    """
    types = SimConfig.field_types()
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or key not in types:
            raise SystemExit(f"bad override {item!r}; known keys: {', '.join(types)}")
        result[key] = parse_value(key, raw, types[key])
    return result


def parse_value(key: str, raw: str, kind: type) -> Any:
    """Convert a raw command-line string to the type of a configuration field.

    :param key: field name, used in the error message.
    :param raw: value as typed by the user.
    :param kind: target type of the field.
    :return: the converted value.
    :raises SystemExit: if the value cannot be converted.
    """
    try:
        return kind(raw)
    except ValueError:
        raise SystemExit(f"cannot parse {key}={raw!r} as {kind.__name__}") from None


def run_one(cfg: SimConfig) -> dict[str, Any]:
    """Run one simulation; used as the worker function of a sweep.

    :param cfg: parameters of the run.
    :return: the configuration fields merged with the end-of-run metrics.
    """
    result = Simulation(cfg).run()
    return {**cfg.to_dict(), **result.metrics}


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` command: one simulation, printed as a table or as JSON.

    :param args: parsed command-line arguments.
    :raises SystemExit: on an invalid configuration override.
    """
    cfg = SimConfig(**parse_overrides(args.set))
    logger.info("running %s", cfg)
    result = Simulation(cfg).run(snapshot_every=args.snapshot_every)
    if args.json:
        json.dump(
            {
                "config": cfg.to_dict(),
                "metrics": result.metrics,
                "outcomes": result.outcomes,
                "history": result.history,
            },
            sys.stdout,
            indent=2,
        )
        print()
        return
    for snap in result.history:
        print(
            f"tick {snap['tick']:>6}: |C|={snap['code_size']:<3} d={snap['d']}  records/node={snap['records_mean']:.1f}"
        )
    print("outcomes:", result.outcomes)
    width = max(map(len, result.metrics))
    for key, value in result.metrics.items():
        print(f"{key:<{width}}  {value:.4g}" if isinstance(value, float) else f"{key:<{width}}  {value}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Execute the ``sweep`` command: vary one parameter over several seeds in parallel.

    Prints the mean and standard deviation of the summary metrics for every value and optionally writes one CSV row
    per run.

    :param args: parsed command-line arguments.
    :raises SystemExit: on an unknown parameter or an invalid value.
    """
    base = SimConfig(**parse_overrides(args.set))
    kind = SimConfig.field_types().get(args.param)
    if kind is None:
        raise SystemExit(f"unknown parameter {args.param!r}")
    values = [parse_value(args.param, v, kind) for v in args.values.split(",")]
    configs = [base.with_(**{args.param: v, "seed": base.seed + s}) for v in values for s in range(args.seeds)]
    logger.info("sweeping %s over %s, %d seeds each: %d runs", args.param, values, args.seeds, len(configs))

    with ProcessPoolExecutor(max_workers=args.workers or None) as pool:
        rows = list(pool.map(run_one, configs))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("wrote %d rows to %s", len(rows), path)

    print(f"{args.param:>10} " + " ".join(f"{c:>18}" for c in SUMMARY_COLUMNS))
    for v in values:
        group = [r for r in rows if r[args.param] == v]
        cells = []
        for col in SUMMARY_COLUMNS:
            xs = [r[col] for r in group if r[col] is not None]
            if not xs:
                cells.append(f"{'n/a':>18}")
            else:
                sd = stdev(xs) if len(xs) > 1 else 0.0
                cells.append(f"{mean(xs):>9.3f} ± {sd:<6.3f}")
        print(f"{v:>10} " + " ".join(cells))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the ``run`` and ``sweep`` subcommands.

    :return: the configured parser.
    """
    parser = argparse.ArgumentParser(prog="self-organized-codes", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="single run, print final metrics")
    run.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    run.add_argument("--snapshot-every", type=int, default=0, help="print metrics every K ticks")
    run.add_argument("--json", action="store_true", help="print full result as JSON")
    run.set_defaults(func=cmd_run)

    sweep = sub.add_parser("sweep", help="vary one parameter over several seeds")
    sweep.add_argument("param", help="config field to vary, e.g. noise")
    sweep.add_argument("values", help="comma-separated values, e.g. 0,0.02,0.05")
    sweep.add_argument("--seeds", type=int, default=30, help="runs per value (seeds seed..seed+K-1)")
    sweep.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="base config overrides")
    sweep.add_argument("--out", help="CSV file for per-run rows")
    sweep.add_argument("--workers", type=int, default=0, help="parallel processes (0 = all cores)")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, configure logging and run the selected command.

    :param argv: arguments without the program name; ``None`` means ``sys.argv[1:]``.
    :raises SystemExit: on invalid arguments.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)
