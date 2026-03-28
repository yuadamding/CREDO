"""
Summarize the fixed-split full-LARRY CAPE vs scDiffeq benchmark.

Run with:
  python summarize_larry_full_split_benchmark.py
"""
import json
import os
from pathlib import Path

from larry_full_benchmark_common import get_output_root, save_json, save_text


OUT = get_output_root()


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(x, digits=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main():
    cape = load(OUT / "cape" / "results_cape_full_split.json")
    sdq = load(OUT / "scdiffeq" / "results_scdiffeq_full_split.json")

    t4_cape = cape["sink_pca_by_time"].get("4.0")
    t4_sdq = sdq["sink_pca_by_time"].get("4.0")
    t6_cape = cape["sink_pca_by_time"].get("6.0")
    t6_sdq = sdq["sink_pca_by_time"].get("6.0")

    cape_train_time = cape.get("train_time_s")
    sdq_train_time = sdq.get("train_time_s")

    summary = {
        "output_root": str(OUT),
        "cape_train_time_s": cape_train_time,
        "scdiffeq_train_time_s": sdq_train_time,
        "cape_peak_gpu_mem_mb": cape["peak_gpu_mem_mb"],
        "scdiffeq_peak_gpu_mem_mb": sdq["peak_gpu_mem_mb"],
        "cape_sink_t4": t4_cape,
        "scdiffeq_sink_t4": t4_sdq,
        "cape_sink_t6": t6_cape,
        "scdiffeq_sink_t6": t6_sdq,
        "cape_t6_tv": cape["t6_label_tv_distance"],
        "scdiffeq_t6_tv": sdq["t6_label_tv_distance"],
        "cape_speedup_vs_scdiffeq": (
            sdq_train_time / cape_train_time
            if cape_train_time not in (None, 0) and sdq_train_time is not None
            else None
        ),
    }

    lines = [
        "# Full-LARRY Fixed-Split Benchmark Summary",
        "",
        f"Output root: `{OUT}`",
        "",
        "| Metric | CAPE | scDiffeq |",
        "| --- | ---: | ---: |",
        f"| Train time (s) | {fmt(cape_train_time, 1)} | {fmt(sdq_train_time, 1)} |",
        f"| Eval time (s) | {fmt(cape['eval_time_s'], 1)} | {fmt(sdq['eval_time_s'], 1)} |",
        f"| Peak GPU mem (MB) | {fmt(cape['peak_gpu_mem_mb'], 1)} | {fmt(sdq['peak_gpu_mem_mb'], 1)} |",
        f"| Sinkhorn t=4 (test PCA) | {fmt(t4_cape, 4)} | {fmt(t4_sdq, 4)} |",
        f"| Sinkhorn t=6 (test PCA) | {fmt(t6_cape, 4)} | {fmt(t6_sdq, 4)} |",
        f"| t6 label TV distance | {fmt(cape['t6_label_tv_distance'], 4)} | {fmt(sdq['t6_label_tv_distance'], 4)} |",
        "",
        f"CAPE training speedup vs scDiffeq: {fmt(summary['cape_speedup_vs_scdiffeq'], 2)}x",
        "",
    ]
    md = "\n".join(lines)

    save_json(OUT / "comparison_summary_full_split.json", summary)
    save_text(OUT / "comparison_summary_full_split.md", md)
    print(md)


if __name__ == "__main__":
    main()
