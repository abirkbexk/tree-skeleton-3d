# treeskel – 3D apple-tree skeleton from RGB-D images

MMA3001 *Numerical Methods and Machine Learning* – individual project (Dataset 1: RGB-D apple-tree reconstruction).

**Task.** Given an RGB image (and an aligned depth image) of an apple tree as PNG files, estimate the tree skeleton – trunk and the two primary
branches of a Y-shaped tree – including branch sections hidden behind leaves and fruit, and lift it to a **3D polyline in metres** with branch lengths.
Training data are RGB-D PNGs with matching skeleton-annotation PNGs. Intended use: orchard perception for pruning / thinning / harvesting robots.

## Method in one paragraph
The skeleton is represented by the horizontal position of the left and right branch for every image row (a regression problem with 2·S outputs).
Features are down-sampled RGB + normalised depth; targets are compressed with an SVD/PCA. Seven models are compared on a validation set
(mean baseline, Ridge, decision tree, kNN, SVR, MLP, small CNN). The winner's output is smoothed by an interpolation scheme chosen on validation data
(global polynomial vs smoothing spline), lifted to 3D with a pinhole camera model and robust depth sampling, and branch length is computed by numerical
integration. Every numerical building block (Newton interpolation, finite differences, Euler/Heun/RK4, Simpson/Romberg/Gauss quadrature, ridge solvers)
is verified against known solutions in `tests/`.

| Package module | Content | Unit topic |
|---|---|---|
| `data.py`, `dataset.py`, `synthetic.py` | acquisition, PNG pairing, validation, depth-hole filling, array assembly | W1 data mgmt, W3 2D interpolation |
| `skeleton.py`, `postprocess.py`, `numerics.py` | row representation, splines, Newton polynomials, finite differences, ODE solvers, quadrature | W3, W6, W7 |
| `models.py`, `cnn.py`, `features.py`, `metrics.py` | regression models, CNN, splits, metrics | W5 |
| `geometry.py`, `ode_recon.py` | pinhole lifting; slope-regression + ODE alternative | W6, W7 |
| `experiments.py` | SVD, learning curves, verification, profiling, conditioning, FLOPs | W2 |
| `pipeline.py`, `cli.py` | training, `SkeletonModel`, `extract_skeleton_3d` | – |

## Install
```bash
git clone https://github.com/abirkbexk/tree-skeleton-3d.git && cd tree-skeleton-3d
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                 # add ,cnn for PyTorch
```
Google Colab: open `notebooks/MMA3001_tree_skeleton_colab.ipynb` (installs the package from this repo; use a GPU runtime for the CNN).

## Data
The labelled **Monash TreeID** Summer images (400 apple trees x RGB + depth + ground-truth skeleton = 1200 PNGs, ~250 MB) are **bundled in `data/TreeID/`**, so a clone is
ready to train: the default `DATA_MODE="repo"` finds them automatically and groups files by tree (`GROUP_REGEX=(r\d+n\d+)`) so a tree never straddles train/val/test.
See [`data/README.md`](data/README.md) for the file format and the depth-encoding assumption (8-bit, brighter = nearer, metric scale approximate).
Other sources remain available via `--data-mode`: `local`, `gdown`, `drive`, or `synthetic` (generated trees - **only for testing the pipeline**).

## Usage
```bash
# smoke test in ~1 minute, no data needed
treeskel-train --data-mode synthetic --quick --no-cnn --out-dir outputs

# real data (bundled data/TreeID, default)
treeskel-train --out-dir outputs                                                    # writes outputs/skeleton_model.joblib + tables
# your own folder: treeskel-train --data-mode local --local-path /path/to/pngs
treeskel-predict tree_rgb.png --depth tree_depth.png --model outputs/skeleton_model.joblib --out-dir outputs/predictions

# all report tables/figures
python scripts/run_experiments.py --data-mode local --local-path /path/to/pngs --out-dir outputs
```
```python
from treeskel import extract_skeleton_3d
res = extract_skeleton_3d("tree_rgb.png", "tree_depth.png", model="outputs/skeleton_model.joblib")
res["length_m"], res["points_3d"]["left"]      # metres, camera frame (X right, Y down, Z forward)
```

### Inputs and outputs
| | Format | Notes |
|---|---|---|
| RGB | PNG, 8-bit (any size) | centre-cropped to a square, resized to `IMG_SIZE` (default 256) |
| Depth (optional) | PNG, 16-bit, millimetres (`DEPTH_SCALE_M`) | 0 or > `DEPTH_MAX_M` = missing; 8-bit depth is accepted with a warning (metric scale lost) |
| Skeleton annotation (training only) | PNG, bright line on dark background (inverted automatically) | one run of pixels per branch per row |
| **Output** | dict: `rows_px` (2, m) branch x-positions; `points_3d` (m, 3) per branch in metres; `length_m`; `junction_row`; `extent`; CSV and PLY export | |

Invalid input: missing file → `FileNotFoundError`; non-PNG → `ValueError`; unreadable/empty annotation → sample skipped and reported during training;
missing/mostly-empty depth → 2D result only, with a warning.

## Results on the bundled TreeID data
Tree-grouped split (279 train / 61 val / 60 test trees), no CNN; error is skeleton x-position RMSE in pixels of the 256 px working image.
Full tables: [`docs/results/`](docs/results/).

| Model | Val RMSE | Test RMSE | Chamfer | Predict ms/img |
|---|---|---|---|---|
| SVR | 13.54 | 13.58 | 10.05 | 0.67 |
| Ridge (selected on validation) | 13.35 | 14.15 | 10.53 | 0.07 |
| kNN | 14.60 | 14.65 | 10.91 | 0.11 |
| Decision tree | 15.71 | 14.76 | 10.96 | 0.06 |
| Mean-skeleton baseline | 15.76 | 16.04 | 12.23 | 0.01 |
| MLP | 18.47 | 18.07 | 13.12 | 0.06 |

The learned models improve only modestly (about 2 px) on the mean-skeleton baseline, and post-processing (polynomial / smoothing spline) gave no gain over raw predictions on validation data.
Reproduce with `treeskel-train --no-cnn --out-dir outputs`.

## Reproducibility
Fixed seeds (`SEED`), a single configuration dict (`treeskel/config.py`), model selection on validation data only, test set scored once for the final table.
Set `GROUP_REGEX` so that images of the same tree stay in the same split – otherwise a random image split can leak near-duplicate frames and inflate accuracy.

## Tests, test report and documentation
```bash
pytest --cov=treeskel --html=docs/test_report.html --self-contained-html     # tests + coverage + HTML report
pdoc treeskel -o docs/api                                                    # HTML API documentation
```
Committed outputs: [`docs/test_report.html`](docs/test_report.html) and [`docs/api/`](docs/api/index.html). CI (`.github/workflows/ci.yml`) re-runs both on every push.

## Known limitations
* One x-value per branch per image row: near-horizontal branches and trees with more than two primary branches cannot be represented.
* Depth on a hidden branch is the depth of the occluding leaf; robust sampling and smoothing reduce but do not remove this bias. There is no independent 3D ground truth,
  so 3D accuracy is measured against the annotated skeleton lifted with the same depth map.
* Camera intrinsics are nominal Intel RealSense D435 values unless replaced in the config.
* The CNN could not be exercised in the automated test environment used to write this repo (no PyTorch there); `tests/test_cnn.py` runs wherever PyTorch is installed.

## AI use
The required AI-use disclosure and reflection is Section 7 of the written project documentation ([`docs/report/MMA3001_Written_Project_Documentation.docx`](docs/report/MMA3001_Written_Project_Documentation.docx)).

## Licence and references
MIT (see `LICENSE`). Dataset licence: check the dataset page before redistributing any data.
Chen, Z., Granland, K., Newbury, R., Chen, C. (2022). *HOB-CNN: Hallucination of Occluded Branches with a Convolutional Neural Network for 2D Fruit Trees.* arXiv:2208.00002.
