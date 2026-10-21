# Data

Raw images are **not** committed. Obtain the dataset and either let the code download it or point it at a local folder.

* **Dataset:** Monash *Y-shaped Tree Dataset for Tree Branch Prediction* (Pink Lady apple trees, RGB-D, winter/autumn/summer, with position / visible-branch / whole-branch annotations) –
  described at <https://www.monash.edu/engineering/lmga/news/y-shaped-tree-dataset-for-tree-branch-prediction> and in Chen et al. (2022), arXiv:2208.00002.
* **Download:** `treeskel-train --data-mode gdown` uses `GDRIVE_FILE_ID` in `src/treeskel/config.py`. **Verify this is the same link as "Dataset 1" in the unit slides** and change it if not.
* **Local copy:** `--data-mode local --local-path <folder>`; any folder layout works as long as file or folder names contain `rgb`/`color`, `depth`, and
  `skeleton`/`label`/`mask`/`whole`/`position`/`gt` so that files can be paired. Check the printed summary ("N usable samples … skeleton variants seen") and
  use `--skeleton-variant whole` (or `visible`) if several annotation images exist per tree.
* **Seasons** are read from `summer` / `autumn` / `winter` in the path; images without it are treated as `unknown`.
* Respect the dataset licence – do not push images to GitHub.

## Notes on the Monash TreeID folder layout (checked on real files)
- Files are named `R{row}{N|U}{tree}_{Summer|Winter}_{RGB|D|GT}-{h}-{m}-{s}.png`; the loader pairs them by that stem.
- RGB is 480x640x3; depth is an **8-bit** 3-channel PNG (0 = no reading, brighter = nearer, metric scale not stored, so 3D lengths are approximate); GT is a **256x256** binary skeleton of the **centre-square crop** of the image.
- GT exists only for Summer `N` trees; Winter and `U` trees have none (usable for inference only).
- Use `--group-regex '(r\d+n\d+)'` so frames of one tree never straddle train/val/test.
- Example: `treeskel-train --data-mode local --local-path <TreeID folder> --group-regex '(r\d+n\d+)'`
