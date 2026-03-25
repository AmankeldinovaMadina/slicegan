#!/usr/bin/env python3
"""Aggregate per-model metric JSON files into a summary table and figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    out_dir = Path("outputs")
    files = sorted(out_dir.glob("*_metrics.json"))
    if not files:
        raise SystemExit("No metric JSON files found in outputs/")

    rows = []
    box_data = {
        "vf_percent_error": [],
        "two_point_mae": [],
        "nsd_percent_error": [],
        "labels": [],
    }
    skipped = []
    for f in files:
        data = json.loads(f.read_text())
        if not data.get("technical_validation_applicable", True):
            skipped.append(data.get("model_id", f.stem))
            continue
        m = data["metrics"]
        vf_gen = m.get("volume_fraction_generated_mean", m["volume_fraction_generated"])
        vf_err = m.get("volume_fraction_abs_error_mean", m["volume_fraction_abs_error"])
        vf_err_std = m.get("volume_fraction_abs_error_std", float("nan"))
        vf_pct = m.get("volume_fraction_percent_error_mean", float("nan"))
        vf_pct_std = m.get("volume_fraction_percent_error_std", float("nan"))
        s2_err = m.get("two_point_mae_mean", m["two_point_mae"])
        s2_err_std = m.get("two_point_mae_std", float("nan"))
        nsd_err = m.get("normalized_surface_density_abs_error_mean", float("nan"))
        nsd_err_std = m.get("normalized_surface_density_abs_error_std", float("nan"))
        nsd_pct = m.get("normalized_surface_density_percent_error_mean", float("nan"))
        nsd_pct_std = m.get("normalized_surface_density_percent_error_std", float("nan"))
        rows.append(
            {
                "model_id": data["model_id"],
                "vf_abs_error": vf_err,
                "vf_abs_error_std": vf_err_std,
                "vf_percent_error": vf_pct,
                "vf_percent_error_std": vf_pct_std,
                "two_point_mae": s2_err,
                "two_point_mae_std": s2_err_std,
                "nsd_abs_error": nsd_err,
                "nsd_abs_error_std": nsd_err_std,
                "nsd_percent_error": nsd_pct,
                "nsd_percent_error_std": nsd_pct_std,
                "vf_original": m["volume_fraction_original"],
                "vf_generated": vf_gen,
            }
        )

        samples = m.get("slice_metric_samples", {})
        box_data["vf_percent_error"].append(
            samples.get("volume_fraction_percent_error", [vf_pct])
        )
        box_data["two_point_mae"].append(samples.get("two_point_mae", [s2_err]))
        box_data["nsd_percent_error"].append(
            samples.get("normalized_surface_density_percent_error", [nsd_pct])
        )
        box_data["labels"].append(data["model_id"])

    if not rows:
        raise SystemExit("No n-phase metric JSON files found in outputs/")

    # Save CSV summary
    csv_path = out_dir / "metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "model_id,vf_original,vf_generated,vf_abs_error,vf_abs_error_std,"
            "vf_percent_error,vf_percent_error_std,two_point_mae,two_point_mae_std,"
            "nsd_abs_error,nsd_abs_error_std,nsd_percent_error,nsd_percent_error_std\n"
        )
        for r in rows:
            fh.write(
                f"{r['model_id']},{r['vf_original']:.6f},{r['vf_generated']:.6f},"
                f"{r['vf_abs_error']:.6f},{r['vf_abs_error_std']:.6f},"
                f"{r['vf_percent_error']:.6f},{r['vf_percent_error_std']:.6f},"
                f"{r['two_point_mae']:.6f},{r['two_point_mae_std']:.6f},"
                f"{r['nsd_abs_error']:.6f},{r['nsd_abs_error_std']:.6f},"
                f"{r['nsd_percent_error']:.6f},{r['nsd_percent_error_std']:.6f}\n"
            )

    # Plot summary
    model_ids = [r["model_id"] for r in rows]
    vf_pct = [r["vf_percent_error"] for r in rows]
    vf_pct_std = [r["vf_percent_error_std"] for r in rows]
    s2_err = [r["two_point_mae"] for r in rows]
    s2_err_std = [r["two_point_mae_std"] for r in rows]
    nsd_pct = [r["nsd_percent_error"] for r in rows]
    nsd_pct_std = [r["nsd_percent_error_std"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].bar(model_ids, vf_pct, yerr=vf_pct_std, capsize=4)
    ax[0].set_title("Volume Fraction Percent Error")
    ax[0].set_ylabel("|VF_2D - VF_3D| / VF_2D (%)")
    ax[0].tick_params(axis="x", rotation=25)

    ax[1].bar(model_ids, s2_err, yerr=s2_err_std, capsize=4)
    ax[1].set_title("Two-Point Correlation MAE")
    ax[1].set_ylabel("Mean |S2_original - S2_generated|")
    ax[1].tick_params(axis="x", rotation=25)

    ax[2].bar(model_ids, nsd_pct, yerr=nsd_pct_std, capsize=4)
    ax[2].set_title("Surface Density Percent Error")
    ax[2].set_ylabel("|NSD_2D - NSD_3D| / NSD_2D (%)")
    ax[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=180)

    # Distribution view: boxplots over slice-level samples.
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.8))
    ax2[0].boxplot(box_data["vf_percent_error"], tick_labels=box_data["labels"])
    ax2[0].set_title("VF Percent Error Distribution")
    ax2[0].set_ylabel("|VF_2D - VF_3D| / VF_2D (%)")
    ax2[0].tick_params(axis="x", rotation=25)

    ax2[1].boxplot(box_data["two_point_mae"], tick_labels=box_data["labels"])
    ax2[1].set_title("Two-Point MAE Distribution")
    ax2[1].set_ylabel("Mean |S2_original - S2_generated|")
    ax2[1].tick_params(axis="x", rotation=25)

    ax2[2].boxplot(box_data["nsd_percent_error"], tick_labels=box_data["labels"])
    ax2[2].set_title("NSD Percent Error Distribution")
    ax2[2].set_ylabel("|NSD_2D - NSD_3D| / NSD_2D (%)")
    ax2[2].tick_params(axis="x", rotation=25)

    fig2.tight_layout()
    fig2.savefig(out_dir / "metrics_summary_boxplots.png", dpi=180)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'metrics_summary.png'}")
    print(f"Wrote {out_dir / 'metrics_summary_boxplots.png'}")
    if skipped:
        print(f"Skipped non-n-phase technical validation: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
