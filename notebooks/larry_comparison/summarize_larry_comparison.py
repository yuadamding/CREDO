"""
Summarize the latest CAPE vs scDiffeq LARRY benchmark results.

Run with:
  python summarize_larry_comparison.py
or
  LARRY_COMPARISON_OUT=/path/to/out python summarize_larry_comparison.py
"""
import json
import os
from pathlib import Path


OUT = Path(
    os.environ.get(
        "LARRY_COMPARISON_OUT",
        "/home/yding1995/opscc_sc/CAPE/outputs/larry_comparison",
    )
)


def load_json(name: str) -> dict:
    with open(OUT / name) as f:
        return json.load(f)


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    cape = load_json("results_cape.json")
    scdiffeq = load_json("results_scdiffeq.json")

    summary = {
        "output_dir": str(OUT),
        "cape_train_time_s": cape.get("train_time_s"),
        "scdiffeq_train_time_s": scdiffeq.get("train_time_s"),
        "train_speedup_cape_vs_scdiffeq": (
            scdiffeq["train_time_s"] / cape["train_time_s"]
            if cape.get("train_time_s")
            else None
        ),
        "cape_sink_pca_4": cape.get("sink_pca_4"),
        "cape_sink_pca_6": cape.get("sink_pca_6"),
        "scdiffeq_sink_pca_4": scdiffeq.get("sink_pca_4"),
        "scdiffeq_sink_pca_6": scdiffeq.get("sink_pca_6"),
        "cape_peak_gpu_mem_mb": cape.get("peak_gpu_mem_mb"),
        "scdiffeq_peak_gpu_mem_mb": scdiffeq.get("peak_gpu_mem_mb"),
    }

    lines = [
        "# LARRY Benchmark Summary",
        "",
        f"Output directory: `{OUT}`",
        "",
        "| Metric | CAPE | scDiffeq |",
        "| --- | ---: | ---: |",
        f"| Train time (s) | {fmt(cape.get('train_time_s'), 1)} | {fmt(scdiffeq.get('train_time_s'), 1)} |",
        f"| Eval time (s) | {fmt(cape.get('eval_time_s'), 1)} | {fmt(scdiffeq.get('eval_time_s'), 1)} |",
        f"| Fate sim time (s) | {fmt(cape.get('fate_eval_time_s'), 1)} | {fmt(scdiffeq.get('fate_eval_time_s'), 1)} |",
        f"| Peak GPU mem (MB) | {fmt(cape.get('peak_gpu_mem_mb'), 1)} | {fmt(scdiffeq.get('peak_gpu_mem_mb'), 1)} |",
        f"| Sinkhorn t=4 (PCA) | {fmt(cape.get('sink_pca_4'), 4)} | {fmt(scdiffeq.get('sink_pca_4'), 4)} |",
        f"| Sinkhorn t=6 (PCA) | {fmt(cape.get('sink_pca_6'), 4)} | {fmt(scdiffeq.get('sink_pca_6'), 4)} |",
        f"| Fate mono frac | {fmt(cape.get('fate_mono_frac'), 4)} | {fmt(scdiffeq.get('fate_mono_frac'), 4)} |",
        "",
        f"CAPE training speedup vs scDiffeq: {fmt(summary['train_speedup_cape_vs_scdiffeq'], 2)}x",
        "",
    ]

    summary_md = "\n".join(lines)

    with open(OUT / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "comparison_summary.md", "w") as f:
        f.write(summary_md)

    print(summary_md)


if __name__ == "__main__":
    main()
