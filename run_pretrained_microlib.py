from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image

try:
    import plotly.graph_objects as go
    from skimage import measure

    HAS_3D_RENDER_DEPS = True
except Exception:
    HAS_3D_RENDER_DEPS = False

# Import the Microlib SliceGAN architecture helper.
REPO_ROOT = Path(__file__).resolve().parent
MICROLIB_ROOT = REPO_ROOT / "microlib"
if str(MICROLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(MICROLIB_ROOT))

from src.slicegan import networks  # noqa: E402


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_image_type(gf: list[int]) -> str:
    channels_out = gf[-1]
    if channels_out == 1:
        return "grayscale"
    if channels_out == 2:
        return "twophase"
    if channels_out == 3:
        return "threephase"
    return "grayscale"


def postprocess_volume(raw: torch.Tensor, image_type: str) -> np.ndarray:
    raw_np = raw.detach().cpu().numpy()[0]
    if image_type in {"twophase", "threephase"}:
        labels = np.argmax(raw_np, axis=0).astype(np.uint8)
        return labels
    gray = np.clip(raw_np[0], 0.0, 1.0)
    return (gray * 255.0).astype(np.uint8)


def otsu_threshold_uint8(img: np.ndarray) -> int:
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0.0] = np.nan
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom
    idx = int(np.nanargmax(sigma_b2))
    return idx


def to_binary(
    img: np.ndarray, threshold_method: str = "otsu", threshold_value: int | None = None
) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("Expected 2D image for binary conversion")
    uniq = np.unique(img)
    if np.array_equal(uniq, np.array([0, 1])):
        return img.astype(np.uint8)
    if np.array_equal(uniq, np.array([0])) or np.array_equal(uniq, np.array([1])):
        return img.astype(np.uint8)
    if img.dtype != np.uint8:
        work = img.astype(np.float32)
        work = (255.0 * (work - work.min()) / (work.ptp() + 1e-8)).astype(np.uint8)
    else:
        work = img

    if threshold_method == "fixed":
        if threshold_value is None:
            raise ValueError("threshold_value is required when threshold_method=fixed")
        thr = int(np.clip(threshold_value, 0, 255))
    else:
        thr = otsu_threshold_uint8(work)

    return (work >= thr).astype(np.uint8)


def radial_two_point(
    binary_img: np.ndarray, max_r: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    i = binary_img.astype(np.float32)
    h, w = i.shape
    f = np.fft.fft2(i)
    ac = np.fft.ifft2(np.abs(f) ** 2).real / (h * w)
    ac = np.fft.fftshift(ac)

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r_vals = np.arange(0, max_r + 1)
    s2 = np.zeros_like(r_vals, dtype=np.float64)

    for r in r_vals:
        mask = (rr >= r - 0.5) & (rr < r + 0.5)
        if np.any(mask):
            s2[r] = float(np.mean(ac[mask]))
        else:
            s2[r] = np.nan

    return r_vals, s2


def normalized_surface_density_2d(binary_img: np.ndarray) -> float:
    """2D proxy for interface density: normalized phase boundary length per pixel."""
    b = binary_img.astype(np.int16)
    h_trans = np.abs(np.diff(b, axis=1)).sum()
    v_trans = np.abs(np.diff(b, axis=0)).sum()
    return float((h_trans + v_trans) / b.size)


def percent_error(reference: float, value: float) -> float:
    if abs(reference) < 1e-12:
        return float("nan")
    return float(abs(value - reference) / abs(reference) * 100.0)


def compare_slice_metrics(
    original_bin: np.ndarray,
    generated_bin: np.ndarray,
) -> Dict[str, Any]:
    """Compare one generated slice to reference image and return a metric bundle."""

    max_r = int(
        min(
            original_bin.shape[0],
            original_bin.shape[1],
            generated_bin.shape[0],
            generated_bin.shape[1],
        )
        // 3
    )
    r_orig, s2_orig = radial_two_point(original_bin, max_r=max_r)
    sv_orig = normalized_surface_density_2d(original_bin)
    vf_orig = float(original_bin.mean())

    vf_gen = float(generated_bin.mean())
    _, s2_gen = radial_two_point(generated_bin, max_r=max_r)
    sv_gen = normalized_surface_density_2d(generated_bin)
    vf_err = abs(vf_orig - vf_gen)
    s2_err = float(np.nanmean(np.abs(s2_orig - s2_gen)))
    sv_err = abs(sv_orig - sv_gen)
    return {
        "vf_original": vf_orig,
        "vf_generated": vf_gen,
        "vf_abs_error": vf_err,
        "vf_percent_error": percent_error(vf_orig, vf_gen),
        "sv_original": sv_orig,
        "sv_generated": sv_gen,
        "sv_abs_error": sv_err,
        "sv_percent_error": percent_error(sv_orig, sv_gen),
        "s2_original": s2_orig,
        "s2_generated": s2_gen,
        "s2_mae": s2_err,
        "r": r_orig,
    }


def evenly_spaced_indices(length: int, n_samples: int) -> np.ndarray:
    n = max(1, min(length, n_samples))
    idx = np.linspace(0, length - 1, n, dtype=int)
    return np.unique(idx)


def random_indices(length: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = max(1, min(length, n_samples))
    idx = rng.choice(length, size=n, replace=False)
    return np.sort(idx)


def collect_oriented_slices(
    volume: np.ndarray,
    slices_per_axis: int,
    rng: np.random.Generator,
    sample_mode: str,
) -> List[Dict[str, Any]]:
    if volume.ndim != 3:
        raise ValueError("Expected a 3D generated volume")

    out: List[Dict[str, Any]] = []
    for axis in [0, 1, 2]:
        if sample_mode == "random":
            idxs = random_indices(volume.shape[axis], slices_per_axis, rng)
        else:
            idxs = evenly_spaced_indices(volume.shape[axis], slices_per_axis)
        for i in idxs:
            if axis == 0:
                sl = volume[i, :, :]
            elif axis == 1:
                sl = volume[:, i, :]
            else:
                sl = volume[:, :, i]
            out.append({"axis": axis, "index": int(i), "slice": sl})
    return out


def aggregate_metric_values(
    rows: List[Dict[str, Any]], key: str
) -> Tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals))


def to_binary_volume(
    vol: np.ndarray, threshold_method: str = "otsu", threshold_value: int | None = None
) -> np.ndarray:
    if vol.ndim != 3:
        raise ValueError("Expected 3D volume for binary conversion")
    uniq = np.unique(vol)
    if np.array_equal(uniq, np.array([0, 1])):
        return vol.astype(np.uint8)
    if np.array_equal(uniq, np.array([0])) or np.array_equal(uniq, np.array([1])):
        return vol.astype(np.uint8)
    if vol.dtype != np.uint8:
        work = vol.astype(np.float32)
        work = (255.0 * (work - work.min()) / (work.ptp() + 1e-8)).astype(np.uint8)
    else:
        work = vol

    if threshold_method == "fixed":
        if threshold_value is None:
            raise ValueError("threshold_value is required when threshold_method=fixed")
        thr = int(np.clip(threshold_value, 0, 255))
    else:
        thr = otsu_threshold_uint8(work)
    return (work >= thr).astype(np.uint8)


def normalized_surface_density_3d(binary_vol: np.ndarray) -> float:
    """3D interface-density proxy via voxel-face transitions normalized by volume."""
    b = binary_vol.astype(np.int16)
    t0 = np.abs(np.diff(b, axis=0)).sum()
    t1 = np.abs(np.diff(b, axis=1)).sum()
    t2 = np.abs(np.diff(b, axis=2)).sum()
    return float((t0 + t1 + t2) / b.size)


def download_original_png(model_id: str, out_dir: Path) -> Path:
    base = f"https://microstructure-library.s3.eu-west-2.amazonaws.com/{model_id}"
    candidates = [
        f"{base}/{model_id}_original.png",
        f"{base}/{model_id}_inpainted.png",
    ]
    out_path = out_dir / f"{model_id}_original.png"
    for url in candidates:
        try:
            urlretrieve(url, out_path)
            return out_path
        except (HTTPError, URLError):
            continue
    raise RuntimeError(
        f"Could not download original image for {model_id} from microlib S3 URLs"
    )


def load_gray_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def save_side_by_side_html(
    volume: np.ndarray,
    original_png: Path,
    out_html: Path,
    title: str,
    threshold_method: str,
    threshold_value: int | None,
) -> bool:
    if not HAS_3D_RENDER_DEPS:
        return False

    binary = to_binary_volume(
        volume, threshold_method=threshold_method, threshold_value=threshold_value
    )
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
        name="Isosurface",
        showscale=False,
    )

    img = Image.open(original_png).convert("L")
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    encoded = base64.b64encode(buff.getvalue()).decode("ascii")
    img_uri = f"data:image/png;base64,{encoded}"

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=55, b=10),
        scene=dict(
            domain=dict(x=[0.38, 1.0], y=[0.0, 1.0]),
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        images=[
            dict(
                source=img_uri,
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.0,
                sizex=0.34,
                sizey=0.9,
                xanchor="left",
                yanchor="top",
                layer="above",
            )
        ],
        annotations=[
            dict(
                text=f"Original 2D: {original_png.name}",
                x=0.17,
                y=1.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13),
            ),
            dict(
                text="Generated 3D Isosurface",
                x=0.69,
                y=1.02,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=13),
            ),
        ],
    )
    fig.write_html(str(out_html), include_plotlyjs=True)
    return True


def find_model_ids(root: Path) -> list[str]:
    gens = {p.name.replace("_Gen.pt", "") for p in root.glob("*_Gen.pt")}
    pars = {p.name.replace("_params.data", "") for p in root.glob("*_params.data")}
    return sorted(gens.intersection(pars))


def run(
    model_id: str,
    lf: int,
    seeds: List[int],
    out_dir: Path,
    slices_per_axis: int,
    slice_sample_mode: str,
    threshold_method: str,
    threshold_value: int | None,
    allow_phase_inversion: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = REPO_ROOT / model_id
    gen_path = REPO_ROOT / f"{model_id}_Gen.pt"
    params_path = REPO_ROOT / f"{model_id}_params.data"

    if not gen_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"Missing files for {model_id}: expected {gen_path.name} and {params_path.name}"
        )

    with open(params_path, "rb") as fh:
        dk, ds, df, dp, gk, gs, gf, gp = pickle.load(fh)

    image_type = infer_image_type(gf)
    device = select_device()

    _, Gen = networks.slicegan_nets(
        str(model_prefix), False, image_type, dk, ds, df, dp, gk, gs, gf, gp
    )
    netG = Gen().to(device)

    state = torch.load(gen_path, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    original_png = download_original_png(model_id, out_dir)
    original = load_gray_png(original_png)

    # Use one reference 2D image and compare with many generated slices/orientations.
    original_bin = to_binary(
        original, threshold_method=threshold_method, threshold_value=threshold_value
    )
    vf_orig_ref = float(original_bin.mean())
    sv_orig_ref = float(normalized_surface_density_2d(original_bin))

    z_channels = int(gf[0])
    effective_phase_inversion = allow_phase_inversion and image_type == "twophase"

    if image_type == "grayscale":
        print(
            f"Skipping technical validation for {model_id}: paper protocol applies these metrics to n-phase only"
        )
        noise_seed = seeds[0]
        torch.manual_seed(noise_seed)
        noise = torch.randn(1, z_channels, lf, lf, lf, device=device)
        with torch.no_grad():
            raw = netG(noise)
        vol = postprocess_volume(raw, image_type)
        tif_path = out_dir / f"{model_id}_generated_lf{lf}_seed{noise_seed}.tif"
        tifffile.imwrite(tif_path, vol)
        html_path = tif_path.with_name(f"{tif_path.stem}_3d.html")
        html_ok = save_side_by_side_html(
            volume=vol,
            original_png=original_png,
            out_html=html_path,
            title=f"{model_id} | seed={noise_seed}",
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )

        metrics = {
            "model_id": model_id,
            "device": str(device),
            "image_type": image_type,
            "latent_factor_lf": lf,
            "seeds": seeds,
            "technical_validation_applicable": False,
            "technical_validation_skipped_reason": "Paper technical validation metrics are defined for n-phase materials, not grayscale.",
            "generator_files": {
                "generator_weights": str(gen_path),
                "params_file": str(params_path),
                "model_prefix": str(model_prefix),
            },
            "generated_tifs": [str(tif_path)],
            "generated_htmls": [str(html_path)] if html_ok else [],
            "generated_volume_shapes": [list(vol.shape)],
            "original_png": str(original_png),
        }

        metrics_path = out_dir / f"{model_id}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(json.dumps(metrics, indent=2))
        return

    comparisons: List[Dict[str, Any]] = []
    generated_tifs: List[str] = []
    generated_htmls: List[str] = []
    per_seed_summary: List[Dict[str, Any]] = []
    volume_shapes: List[List[int]] = []
    phase_mapping_counts = {"generated": 0, "inverted_generated": 0}

    for seed in seeds:
        torch.manual_seed(seed)
        noise = torch.randn(1, z_channels, lf, lf, lf, device=device)
        with torch.no_grad():
            raw = netG(noise)
        vol = postprocess_volume(raw, image_type)

        if vol.ndim != 3:
            raise RuntimeError("Generated output is expected to be a 3D volume")

        volume_shapes.append(list(vol.shape))
        if len(seeds) == 1:
            tif_path = out_dir / f"{model_id}_generated_lf{lf}.tif"
        else:
            tif_path = out_dir / f"{model_id}_generated_lf{lf}_seed{seed}.tif"
        tifffile.imwrite(tif_path, vol)
        generated_tifs.append(str(tif_path))
        html_path = tif_path.with_name(f"{tif_path.stem}_3d.html")
        html_ok = save_side_by_side_html(
            volume=vol,
            original_png=original_png,
            out_html=html_path,
            title=f"{model_id} | seed={seed}",
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )
        if html_ok:
            generated_htmls.append(str(html_path))

        rng = np.random.default_rng(seed)
        slice_rows = collect_oriented_slices(
            vol,
            slices_per_axis=slices_per_axis,
            rng=rng,
            sample_mode=slice_sample_mode,
        )

        def evaluate_seed_rows(invert: bool) -> List[Dict[str, Any]]:
            seed_rows: List[Dict[str, Any]] = []
            for row in slice_rows:
                generated_bin = to_binary(
                    row["slice"],
                    threshold_method=threshold_method,
                    threshold_value=threshold_value,
                )
                if invert:
                    generated_bin = 1 - generated_bin
                cmp_row = compare_slice_metrics(
                    original_bin=original_bin,
                    generated_bin=generated_bin,
                )
                cmp_row["axis"] = row["axis"]
                cmp_row["index"] = row["index"]
                cmp_row["seed"] = seed
                cmp_row["mapping"] = "inverted_generated" if invert else "generated"
                seed_rows.append(cmp_row)
            return seed_rows

        seed_comparisons = evaluate_seed_rows(invert=False)
        if effective_phase_inversion:
            inv_comparisons = evaluate_seed_rows(invert=True)
            s2_raw, _ = aggregate_metric_values(seed_comparisons, "s2_mae")
            s2_inv, _ = aggregate_metric_values(inv_comparisons, "s2_mae")
            if s2_inv < s2_raw:
                seed_comparisons = inv_comparisons

        if not seed_comparisons:
            raise RuntimeError(f"No slices were evaluated for seed={seed}")

        used_mapping = str(seed_comparisons[0]["mapping"])
        phase_mapping_counts[used_mapping] += len(seed_comparisons)

        vol_bin = to_binary_volume(
            vol,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
        )
        vf3d = float(vol_bin.mean())
        nsd3d = normalized_surface_density_3d(vol_bin)

        vf_err_mean, _ = aggregate_metric_values(seed_comparisons, "vf_abs_error")
        vf_pct_mean, _ = aggregate_metric_values(seed_comparisons, "vf_percent_error")
        sv_err_mean, _ = aggregate_metric_values(seed_comparisons, "sv_abs_error")
        sv_pct_mean, _ = aggregate_metric_values(seed_comparisons, "sv_percent_error")
        s2_mae_mean, _ = aggregate_metric_values(seed_comparisons, "s2_mae")

        vf3d_abs_err = abs(vf_orig_ref - vf3d)
        sv3d_abs_err = abs(sv_orig_ref - nsd3d)

        per_seed_summary.append(
            {
                "seed": seed,
                "generated_tif": str(tif_path),
                "volume_shape": list(vol.shape),
                "volume_fraction_3d_generated": vf3d,
                "normalized_surface_density_3d_generated": nsd3d,
                "volume_fraction_3d_abs_error": vf3d_abs_err,
                "volume_fraction_3d_percent_error": percent_error(vf_orig_ref, vf3d),
                "normalized_surface_density_3d_abs_error": sv3d_abs_err,
                "normalized_surface_density_3d_percent_error": percent_error(
                    sv_orig_ref, nsd3d
                ),
                "phase_mapping_used": used_mapping,
                "slice_metric_means": {
                    "volume_fraction_abs_error": vf_err_mean,
                    "volume_fraction_percent_error": vf_pct_mean,
                    "normalized_surface_density_abs_error": sv_err_mean,
                    "normalized_surface_density_percent_error": sv_pct_mean,
                    "two_point_mae": s2_mae_mean,
                },
            }
        )
        comparisons.extend(seed_comparisons)

    if not comparisons:
        raise RuntimeError("No slices were evaluated")

    best = min(comparisons, key=lambda x: float(x["s2_mae"]))
    vf_orig = float(best["vf_original"])
    sv_orig = float(best["sv_original"])

    vf_gen_mean = float(
        np.mean([r["volume_fraction_3d_generated"] for r in per_seed_summary])
    )
    vf_gen_std = float(
        np.std([r["volume_fraction_3d_generated"] for r in per_seed_summary])
    )
    vf_err_mean = float(
        np.mean([r["volume_fraction_3d_abs_error"] for r in per_seed_summary])
    )
    vf_err_std = float(
        np.std([r["volume_fraction_3d_abs_error"] for r in per_seed_summary])
    )
    vf_pct_mean = float(
        np.mean([r["volume_fraction_3d_percent_error"] for r in per_seed_summary])
    )
    vf_pct_std = float(
        np.std([r["volume_fraction_3d_percent_error"] for r in per_seed_summary])
    )

    sv_gen_mean = float(
        np.mean(
            [r["normalized_surface_density_3d_generated"] for r in per_seed_summary]
        )
    )
    sv_gen_std = float(
        np.std([r["normalized_surface_density_3d_generated"] for r in per_seed_summary])
    )
    sv_err_mean = float(
        np.mean(
            [r["normalized_surface_density_3d_abs_error"] for r in per_seed_summary]
        )
    )
    sv_err_std = float(
        np.std([r["normalized_surface_density_3d_abs_error"] for r in per_seed_summary])
    )
    sv_pct_mean = float(
        np.mean(
            [r["normalized_surface_density_3d_percent_error"] for r in per_seed_summary]
        )
    )
    sv_pct_std = float(
        np.std(
            [r["normalized_surface_density_3d_percent_error"] for r in per_seed_summary]
        )
    )

    s2_mae_mean, s2_mae_std = aggregate_metric_values(comparisons, "s2_mae")

    s2_gen_stack = np.stack([c["s2_generated"] for c in comparisons], axis=0)
    s2_gen_mean = np.nanmean(s2_gen_stack, axis=0)
    r_vals = best["r"]
    s2_orig = best["s2_original"]

    rep_seed = int(best["seed"])
    rep_tif = next(
        (Path(t) for t in generated_tifs if f"seed{rep_seed}" in t),
        Path(generated_tifs[0]),
    )
    rep_vol = tifffile.imread(rep_tif)
    rep_axis = int(best["axis"])
    rep_idx = int(best["index"])
    if rep_axis == 0:
        representative_slice = rep_vol[rep_idx, :, :]
    elif rep_axis == 1:
        representative_slice = rep_vol[:, rep_idx, :]
    else:
        representative_slice = rep_vol[:, :, rep_idx]

    fig = plt.figure(figsize=(14, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original 2D Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(representative_slice, cmap="gray")
    ax2.set_title(f"Representative Generated Slice (axis={rep_axis}, idx={rep_idx})")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(["Original", "Generated Mean"], [vf_orig, vf_gen_mean])
    ax3.errorbar(
        [1],
        [vf_gen_mean],
        yerr=[vf_gen_std],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    ax3.set_ylabel("Phase Volume Fraction")
    ax3.set_title(f"VF 3D vs 2D (mean % err={vf_pct_mean:.2f}%, std={vf_pct_std:.2f}%)")
    ax3.set_ylim(0, 1)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(r_vals, s2_orig, label="Original")
    ax4.plot(r_vals, s2_gen_mean, label="Generated Mean over slices")
    s2_gen_std = np.nanstd(s2_gen_stack, axis=0)
    ax4.fill_between(
        r_vals,
        s2_gen_mean - s2_gen_std,
        s2_gen_mean + s2_gen_std,
        alpha=0.2,
        label="Generated +/- 1 std",
    )
    ax4.set_xlabel("Lag distance r (pixels)")
    ax4.set_ylabel("S2(r)")
    ax4.set_title(
        f"Two-Point Correlation (MAE mean={s2_mae_mean:.4f}, std={s2_mae_std:.4f})"
    )
    ax4.legend()

    fig.suptitle(
        f"Microlib Validation: {model_id} | Device: {device} | seeds={len(seeds)} | slices={len(comparisons)}"
    )
    fig.tight_layout()

    plot_path = out_dir / f"{model_id}_metrics.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    metrics = {
        "model_id": model_id,
        "device": str(device),
        "image_type": image_type,
        "latent_factor_lf": lf,
        "seeds": seeds,
        "validation_protocol": {
            "slices_per_axis": slices_per_axis,
            "total_slices_evaluated": len(comparisons),
            "slice_sample_mode": slice_sample_mode,
            "axes_used": [0, 1, 2],
            "threshold_method": threshold_method,
            "threshold_value": threshold_value,
            "allow_phase_inversion": allow_phase_inversion,
            "effective_phase_inversion": effective_phase_inversion,
            "phase_mapping_policy": "per-seed global mapping; no per-slice optimization",
            "technical_validation_applicable": True,
            "paper_alignment": {
                "vf_and_surface_reported_as_percent_error": True,
                "n_phase_only_validation": True,
            },
            "metric_methods": {
                "volume_fraction": "phase_fraction",
                "normalized_surface_density_2d": "pixel-phase-boundary-length-per-pixel",
                "normalized_surface_density_3d": "voxel-face-transition-density-per-voxel",
                "two_point_correlation": "radial-autocorrelation-via-fft",
            },
        },
        "generator_files": {
            "generator_weights": str(gen_path),
            "params_file": str(params_path),
            "model_prefix": str(model_prefix),
        },
        "generated_tifs": generated_tifs,
        "generated_htmls": generated_htmls,
        "generated_volume_shapes": volume_shapes,
        "original_png": str(original_png),
        "per_seed": per_seed_summary,
        "metrics": {
            "volume_fraction_original": vf_orig,
            "volume_fraction_generated_mean": vf_gen_mean,
            "volume_fraction_generated_std": vf_gen_std,
            "volume_fraction_abs_error_mean": vf_err_mean,
            "volume_fraction_abs_error_std": vf_err_std,
            "volume_fraction_percent_error_mean": vf_pct_mean,
            "volume_fraction_percent_error_std": vf_pct_std,
            "normalized_surface_density_original": sv_orig,
            "normalized_surface_density_generated_mean": sv_gen_mean,
            "normalized_surface_density_generated_std": sv_gen_std,
            "normalized_surface_density_abs_error_mean": sv_err_mean,
            "normalized_surface_density_abs_error_std": sv_err_std,
            "normalized_surface_density_percent_error_mean": sv_pct_mean,
            "normalized_surface_density_percent_error_std": sv_pct_std,
            "two_point_mae_mean": s2_mae_mean,
            "two_point_mae_std": s2_mae_std,
            "volume_fraction_3d_generated_mean": float(
                np.mean([r["volume_fraction_3d_generated"] for r in per_seed_summary])
            ),
            "volume_fraction_3d_generated_std": float(
                np.std([r["volume_fraction_3d_generated"] for r in per_seed_summary])
            ),
            "normalized_surface_density_3d_generated_mean": float(
                np.mean(
                    [
                        r["normalized_surface_density_3d_generated"]
                        for r in per_seed_summary
                    ]
                )
            ),
            "normalized_surface_density_3d_generated_std": float(
                np.std(
                    [
                        r["normalized_surface_density_3d_generated"]
                        for r in per_seed_summary
                    ]
                )
            ),
            "slice_metric_samples": {
                "volume_fraction_abs_error": [
                    float(c["vf_abs_error"]) for c in comparisons
                ],
                "volume_fraction_percent_error": [
                    float(c["vf_percent_error"]) for c in comparisons
                ],
                "normalized_surface_density_abs_error": [
                    float(c["sv_abs_error"]) for c in comparisons
                ],
                "normalized_surface_density_percent_error": [
                    float(c["sv_percent_error"]) for c in comparisons
                ],
                "two_point_mae": [float(c["s2_mae"]) for c in comparisons],
            },
            "phase_mapping_counts": phase_mapping_counts,
            "representative_slice": {
                "seed": rep_seed,
                "axis": rep_axis,
                "index": rep_idx,
                "mapping": best["mapping"],
            },
            # Backward-compatible aliases for older summary scripts.
            "volume_fraction_generated": vf_gen_mean,
            "volume_fraction_abs_error": vf_err_mean,
            "two_point_mae": s2_mae_mean,
        },
        "plot": str(plot_path),
    }

    metrics_path = out_dir / f"{model_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretrained Microlib SliceGAN model and compute metrics"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="microstructure376",
        help="Model prefix, e.g. microstructure376",
    )
    parser.add_argument(
        "--lf",
        type=int,
        default=4,
        help="Latent cube edge length (higher gives larger output volume)",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for latent noise"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list, e.g. 7,11,23; overrides --seed",
    )
    parser.add_argument(
        "--slices-per-axis",
        type=int,
        default=10,
        help="Number of generated slices sampled per axis for aggregated validation",
    )
    parser.add_argument(
        "--slice-sample-mode",
        type=str,
        default="random",
        choices=["random", "even"],
        help="How to choose slice indices on each axis",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="otsu",
        choices=["otsu", "fixed"],
        help="Binarization threshold method used before metric computation",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=None,
        help="Fixed threshold value in [0,255] when --threshold-method fixed",
    )
    parser.add_argument(
        "--allow-phase-inversion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional label swap for two-phase data; paper-aligned default is disabled",
    )
    parser.add_argument("--out-dir", type=str, default="outputs", help="Output folder")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available local model ids and exit",
    )
    args = parser.parse_args()

    model_ids = find_model_ids(REPO_ROOT)
    if args.list_models:
        for mid in model_ids:
            print(mid)
        return

    if args.model_id not in model_ids:
        raise ValueError(
            f"Model {args.model_id} not found. Available: {', '.join(model_ids) if model_ids else 'none'}"
        )

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("--seeds must contain at least one integer")
    else:
        seeds = [args.seed]

    run(
        model_id=args.model_id,
        lf=args.lf,
        seeds=seeds,
        out_dir=Path(args.out_dir),
        slices_per_axis=args.slices_per_axis,
        slice_sample_mode=args.slice_sample_mode,
        threshold_method=args.threshold_method,
        threshold_value=args.threshold_value,
        allow_phase_inversion=args.allow_phase_inversion,
    )


if __name__ == "__main__":
    main()
