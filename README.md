

<img width="1435" height="832" alt="Screenshot 2026-03-25 at 15 03 16" src="https://github.com/user-attachments/assets/e4c0afd6-32a5-4f00-a85e-d196d183dcf8" />

## 1) Setup and generation using pretrained models

Used local pretrained files:
- microstructure376_Gen.pt + microstructure376_params.data
- microstructure393_Gen.pt + microstructure393_params.data
- microstructure516_Gen.pt + microstructure516_params.data

Generation/validation script:
- run_pretrained_microlib.py

Summary script:
- summarize_metrics.py

Device selection is automatic and prefers Apple Silicon GPU (`mps`), then CUDA, then CPU.

## 2) Metrics and plots (Original vs Generated)

Implemented metrics (paper-aligned):
- Volume fraction percentage error: |VF_2D - VF_3D| / VF_2D * 100
- Normalized surface density percentage error: |NSD_2D - NSD_3D| / NSD_2D * 100
- Radial two-point correlation MAE: mean |S2_original - S2_generated|

Note:
- Validation is distributional, not single-slice: multiple generated slices are sampled on XY/XZ/YZ and aggregated.
- Technical validation metrics follow the paper's n-phase-only scope; grayscale entries are generated but excluded from aggregate metric plots.
- Phase inversion is disabled by default for paper alignment. If enabled, mapping is chosen globally per seed (not per-slice).
- Surface-area-density is implemented as a voxel-interface-density proxy and reported transparently.

Generated artifacts are in outputs/:
- Per model: *_generated_lf4.tif, *_metrics.json, *_metrics.png, *_original.png
- Combined: metrics_summary.csv, metrics_summary.png


## Commands used

List detected models:

python run_pretrained_microlib.py --list-models

Run one model (single seed):

python run_pretrained_microlib.py --model-id microstructure376 --lf 4 --out-dir outputs

Run one model (paper-aligned protocol):

python run_pretrained_microlib.py --model-id microstructure393 --lf 4 --seeds 7,11,23 --slices-per-axis 10 --slice-sample-mode random --threshold-method otsu --out-dir outputs

Run all local models (paper-aligned protocol):

for m in microstructure376 microstructure393 microstructure516; do
  python run_pretrained_microlib.py --model-id "$m" --lf 4 --seeds 7,11,23 --slices-per-axis 10 --slice-sample-mode random --threshold-method otsu --out-dir outputs
done

Create summary table + chart:

python summarize_metrics.py
