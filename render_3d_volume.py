#!/usr/bin/env python3
"""True 3D volume rendering for generated microstructure TIFF volumes.

This script creates an interactive 3D isosurface rendering and saves it as an
HTML file that can be opened in any browser.

Examples:
  python render_3d_volume.py --tif outputs/microstructure376_generated_lf4.tif
  python render_3d_volume.py --tif outputs/microstructure376_generated_lf4.tif --open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
from skimage import measure


def load_volume(path: Path) -> np.ndarray:
    vol = tifffile.imread(path)
    vol = np.asarray(vol)
    if vol.ndim == 4:
        # Channel-first fallback: keep first channel for rendering.
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")
    return vol


def to_binary(volume: np.ndarray, threshold: float | None) -> np.ndarray:
    v = volume.astype(np.float32)
    if threshold is None:
        # Otsu-like fallback for small discrete ranges.
        uniq = np.unique(v)
        if uniq.size <= 3:
            return (v > np.median(uniq)).astype(np.uint8)
        hist, edges = np.histogram(v, bins=256)
        mids = (edges[:-1] + edges[1:]) * 0.5
        w1 = np.cumsum(hist)
        w2 = np.cumsum(hist[::-1])[::-1]
        m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
        m2 = (np.cumsum((hist * mids)[::-1]) / np.maximum(w2[::-1], 1))[::-1]
        inter = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
        idx = int(np.argmax(inter)) if inter.size else 127
        threshold = float(mids[idx])
    return (v >= float(threshold)).astype(np.uint8)


def build_figure(binary: np.ndarray, title: str) -> go.Figure:
    verts, faces, _, _ = measure.marching_cubes(binary, level=0.5)
    x, y, z = verts[:, 2], verts[:, 1], verts[:, 0]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="#1f77b4",
        opacity=0.55,
        flatshading=False,
        name="Isosurface",
        showscale=False,
        lighting=dict(ambient=0.45, diffuse=0.7, specular=0.2, roughness=0.8),
        lightposition=dict(x=100, y=150, z=200),
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            xaxis=dict(backgroundcolor="rgb(245,245,245)"),
            yaxis=dict(backgroundcolor="rgb(245,245,245)"),
            zaxis=dict(backgroundcolor="rgb(245,245,245)"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render 3D TIFF as interactive isosurface"
    )
    parser.add_argument("--tif", type=str, required=True, help="Path to 3D TIFF")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binarization threshold (optional)",
    )
    parser.add_argument("--out-html", type=str, default=None, help="Output HTML path")
    parser.add_argument(
        "--open", action="store_true", help="Open output HTML in browser"
    )
    args = parser.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    vol = load_volume(tif_path)
    binary = to_binary(vol, args.threshold)

    out_html = (
        Path(args.out_html)
        if args.out_html
        else tif_path.with_suffix("").with_name(tif_path.stem + "_3d.html")
    )

    fig = build_figure(binary, title=f"3D Volume Rendering: {tif_path.name}")
    fig.write_html(str(out_html), include_plotlyjs=True)

    print(f"Input shape: {vol.shape}")
    print(f"Saved 3D render: {out_html}")

    if args.open:
        webbrowser.open(out_html.resolve().as_uri())


if __name__ == "__main__":
    main()
