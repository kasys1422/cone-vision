# Licenses and provenance

## Project code

Original Python and JavaScript implementation in this repository is offered under **GPL-3.0-only**. `LICENSE` contains that license. Official YOLOX model modules are vendored under their original **Apache-2.0** license for an executed detector comparison. There is no imported Ultralytics, CenterNet, DSNT implementation or pretrained checkpoint. Published architectural ideas are cited in `reports/architecture.md`. A concept citation does not imply that this package reproduces a published benchmark performance.

## Provided 3D cone, textures, extracted geometry and synthetic images

- Author: **Mugicha3G / @Mugicha3G**.
- Work: 【Free/無料】 三角コーン 3Dモデル 24種（FBX形式・Blenderファイル付属）.
- Source: <https://mugicha3.booth.pm/items/4001203>.
- License: **Creative Commons Attribution-ShareAlike 3.0 Unported**, <https://creativecommons.org/licenses/by-sa/3.0/>.
- Evidence: supplied archive's unchanged `assets/cone/readme.txt`. The source archive was provided by the user; it is not a newly fetched training dataset.
- Modifications: FBX triangulation and UV/normal extraction for weighted and unweighted source geometry, one uniform scale to a configured 0.700m height, optional horizontal reflective bands, procedural material tint/lighting/scene rendering, camera projection labels, image degradations and prediction overlays. The original source remains attributed and is not relicensed as GPL.
- `assets/cone_mesh.npz`, `assets/cone_geometry.json`, and synthetic image/label collections retain **CC-BY-SA-3.0**, to the extent copyright or similar rights apply. Procedural backgrounds contain no third-party photographs or external textures.
- The reference screenshot is excluded from training and is not redistributed in this project.

## Trained weights

The two networks are trained from random initialization only on this synthetic dataset. There are **no upstream pretrained weights** with separate restrictions.

The distributed `.pt` and `.onnx` weight files are offered as separate model-data artifacts under **CC-BY-SA-3.0**, to the extent rights exist. This is a conservative distribution choice, not a legal conclusion that object-detection weights necessarily adapt the 3D model. Whether a particular trained model contains protected expression, and whether ShareAlike applies, is fact- and jurisdiction-dependent. The independent application and training code retain GPL-3.0-only. Do not describe every file in this mixed code/data distribution as GPL-only or claim that CC-BY-SA 3.0 is automatically identical to GPL-3.0.

Creative Commons' guidance distinguishes copyright applicability from voluntary conservative compliance: <https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/>. The source license does not prohibit commercial use. No additional noncommercial restriction is introduced here.

## Optional real photographs (qualitative checks only)

The complete source URLs, selected licenses, authors, downloaded hashes and modification statements are in `datasets/real_qualitative/sources.json`. These images are excluded from all train/validation/test manifests and from model selection. They have no ground-truth landmark, distance, or camera calibration labels.

| Download | Author | Selected license |
|---|---|---|
| Road-cones,sankaku-cones,japan.JPG | Katorisi | CC BY 2.5 (one of the offered licenses) |
| Traffic cones (698425995).jpg | michell zappa | CC BY-SA 2.0 |
| Osaka traffic cone.jpg | andreas / andyket | CC BY 2.0 |
| Japanese style safety cone. (54190606418).jpg | MIKI Yoshihito | CC BY 2.0 |

Prediction overlays and resized comparison panels are modifications, not original photos. Each panel credits its source; individual overlays preserve the respective image license. A combined contact sheet, if supplied, is attributed with those components and CC-BY-SA-3.0 for new contributions where applicable. None is evidence of real-world quantitative accuracy.

## Runtime dependencies (not vendored into source archive)

| Component | Tested version | Upstream license / source |
|---|---|---|
| PyTorch | 2.7.1+cpu | BSD-style; <https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE> |
| torchvision | 0.22.1+cpu | BSD-3-Clause; <https://github.com/pytorch/vision/blob/v0.22.1/LICENSE> |
| NumPy | 2.3.5 | BSD-3-Clause; <https://numpy.org/doc/stable/license.html> |
| SciPy | 1.17.0 | BSD-3-Clause; <https://scipy.org/about/> |
| OpenCV | 4.11.0 | Apache-2.0; <https://opencv.org/license/> |
| OpenCV.js browser binary | 4.13.0 build from official 4.x distribution | Apache-2.0; pinned binary hash and upstream license accompany web_demo/vendor |
| ONNX | 1.18.0 | Apache-2.0; <https://github.com/onnx/onnx/blob/v1.18.0/LICENSE> |
| ONNX Runtime | 1.22.0 | MIT; <https://github.com/microsoft/onnxruntime/blob/v1.22.0/LICENSE> |
| ONNX Runtime Web | see web_demo/package-lock.json | MIT; same upstream repository |
| Numba / llvmlite | 0.62.1 / 0.45.1 | BSD-2-Clause, with third-party notices; <https://github.com/numba/numba/blob/0.62.1/LICENSE> |
| Pillow | 12.3.0 | HPND; <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| pytest | 8.4.1 | MIT; <https://github.com/pytest-dev/pytest/blob/8.4.1/LICENSE> |
| Matplotlib (optional report figures) | 3.10.8 | PSF-based; <https://matplotlib.org/stable/project/license.html> |

Packages can include third-party notices beyond the top-level license. Installation retrieves those packages with their own notices. Their license files are retained by pip/npm; this repository does not relicense dependency source. The exact installed package versions are captured in `reports/environment.json`.

## Vendored comparison architecture

- YOLOX official model code at commit **6ddff4824372906469a7fae2dc3206c7aa4bbaee** is included under `detector/vendor/yolox/`, with the upstream Apache-2.0 license. Upstream: <https://github.com/Megvii-BaseDetection/YOLOX>. Its architecture is actually trained from random initialization here. No upstream pretrained checkpoint is used. See its provenance notice and training log for the exact scope and local import adjustments.
- The official OpenCV.js compiled browser runtime is also vendored, with its provenance, SHA256 and upstream license. Browser PnP has an executed numerical test; physical-device compatibility remains separate.

## Compared but not imported

- Ultralytics YOLO: AGPL-3.0 / enterprise options (<https://www.ultralytics.com/license>); no code or weights used. A GPL project should not silently relabel AGPL-covered components as GPL-only. Commercial/noncommercial status alone does not decide copyleft obligations.

## V12

Source lineage, COCO initialization and learned weights are documented in [V12 provenance](models/tracon_v12/PROVENANCE.md). Training and evaluation used the existing project's TraCon and synthetic v6 records. Original third-party photographs are excluded from distribution.
