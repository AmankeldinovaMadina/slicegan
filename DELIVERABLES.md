# SliceGAN Task Deliverables (Mac M1)

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

## 3) Video recording script (voice-over)

You can use this exact script while screen sharing:

"In this project I used pretrained SliceGAN generators linked to the microlib dataset from the Nature Scientific Data paper.

I am running on a Mac M1 machine, so I configured the workflow to use PyTorch on Apple Silicon via the MPS backend. The script automatically falls back to CPU if MPS is not available.

I used the provided pretrained generator and parameter pairs, without retraining the model, to generate 3D microstructure volumes as TIFF files.

For validation, I compared each generated sample against the corresponding original 2D image from the microlib dataset. I computed two metrics inspired by the paper’s technical validation approach: phase volume fraction and two-point correlation statistics.

Specifically, I report volume fraction absolute error and radial two-point correlation mean absolute error. Lower values mean generated microstructures better match the original statistics.

The summary plot shows that microstructure393 has the best agreement, microstructure376 is moderate, and microstructure516 is the hardest case with noticeably larger error.

This indicates the pretrained models can reproduce realistic microstructure statistics in many cases, while performance varies by material complexity and image characteristics.

Overall, the task demonstrates successful pretrained SliceGAN deployment, 3D generation, and quantitative comparison between original and generated structures."

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
