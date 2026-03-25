#!/usr/bin/env python3
"""Interactive viewer for 3D TIFF microstructure volumes.

Usage examples:
  python view_3d_tif.py --tif outputs/microstructure376_generated_lf4.tif
  python view_3d_tif.py --tif outputs/microstructure393_generated_lf4.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.widgets import Slider


def load_volume(path: Path) -> np.ndarray:
    vol = tifffile.imread(path)
    vol = np.asarray(vol)

    if vol.ndim == 4:
        # Common case for channel-first volumes; keep first channel for display.
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {vol.shape}")

    return vol


def normalize_for_display(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol - vmin) / (vmax - vmin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D TIFF viewer")
    parser.add_argument("--tif", type=str, required=True, help="Path to 3D .tif volume")
    args = parser.parse_args()

    tif_path = Path(args.tif)
    if not tif_path.exists():
        raise FileNotFoundError(f"File not found: {tif_path}")

    vol = load_volume(tif_path)
    disp = normalize_for_display(vol)

    z_max, y_max, x_max = np.array(disp.shape) - 1
    z0, y0, x0 = z_max // 2, y_max // 2, x_max // 2

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.12])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_mip = fig.add_subplot(gs[1, :])

    ax_z_slider = fig.add_subplot(gs[2, 0])
    ax_y_slider = fig.add_subplot(gs[2, 1])
    ax_x_slider = fig.add_subplot(gs[2, 2])

    im_xy = ax_xy.imshow(disp[z0, :, :], cmap="gray", origin="lower")
    ax_xy.set_title(f"XY (z={z0})")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    im_xz = ax_xz.imshow(disp[:, y0, :], cmap="gray", origin="lower", aspect="auto")
    ax_xz.set_title(f"XZ (y={y0})")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    im_yz = ax_yz.imshow(disp[:, :, x0], cmap="gray", origin="lower", aspect="auto")
    ax_yz.set_title(f"YZ (x={x0})")
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")

    mip = np.max(disp, axis=0)
    im_mip = ax_mip.imshow(mip, cmap="gray", origin="lower")
    ax_mip.set_title("Max Intensity Projection (XY)")
    ax_mip.set_xlabel("x")
    ax_mip.set_ylabel("y")

    z_slider = Slider(ax_z_slider, "z", 0, z_max, valinit=z0, valstep=1)
    y_slider = Slider(ax_y_slider, "y", 0, y_max, valinit=y0, valstep=1)
    x_slider = Slider(ax_x_slider, "x", 0, x_max, valinit=x0, valstep=1)

    def update(_val: float) -> None:
        z = int(z_slider.val)
        y = int(y_slider.val)
        x = int(x_slider.val)

        im_xy.set_data(disp[z, :, :])
        im_xz.set_data(disp[:, y, :])
        im_yz.set_data(disp[:, :, x])

        ax_xy.set_title(f"XY (z={z})")
        ax_xz.set_title(f"XZ (y={y})")
        ax_yz.set_title(f"YZ (x={x})")

        fig.canvas.draw_idle()

    z_slider.on_changed(update)
    y_slider.on_changed(update)
    x_slider.on_changed(update)

    fig.suptitle(
        f"3D Viewer: {tif_path.name} | shape={disp.shape} | "
        "Use sliders, arrow keys, or scroll wheel"
    )

    def on_key(event) -> None:
        if event.key == "up":
            z_slider.set_val(min(z_max, int(z_slider.val) + 1))
        elif event.key == "down":
            z_slider.set_val(max(0, int(z_slider.val) - 1))
        elif event.key == "right":
            x_slider.set_val(min(x_max, int(x_slider.val) + 1))
        elif event.key == "left":
            x_slider.set_val(max(0, int(x_slider.val) - 1))
        elif event.key == "pageup":
            y_slider.set_val(min(y_max, int(y_slider.val) + 1))
        elif event.key == "pagedown":
            y_slider.set_val(max(0, int(y_slider.val) - 1))

    def on_scroll(event) -> None:
        step = 1 if event.button == "up" else -1
        if event.inaxes is ax_xy:
            z_slider.set_val(np.clip(int(z_slider.val) + step, 0, z_max))
        elif event.inaxes is ax_xz:
            y_slider.set_val(np.clip(int(y_slider.val) + step, 0, y_max))
        elif event.inaxes is ax_yz:
            x_slider.set_val(np.clip(int(x_slider.val) + step, 0, x_max))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Avoid unused variable lint warning where strict settings apply.
    _ = im_mip

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
