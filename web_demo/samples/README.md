# Included standalone examples

These raw synthetic images allow the release demo to work without downloading
the full image dataset. They contain no painted predictions, labels or markers.
Use the red/yellow/multiple sample buttons in `web_demo/index.html`: the associated
calibration and simulation dimensions are loaded and inference runs locally.
When a user-uploaded image replaces a sample, sample calibration is cleared so
that its known synthetic K/size cannot accidentally be reused on a real photo.

Each PNG has two companion files:

- `NAME.camera.json`: exact synthetic K, zero lens distortion, image resolution,
  configured0.700m cone height and model base width. These are known scene
  parameters, not a physical measurement of a real cone.
- `NAME.expected.json`: ground-truth annotations and the frozen Python ONNX
  pipeline result. Browser poses can differ slightly because browser refinement
  is unweighted while Python uses weighted robust refinement.

Examples were selected for clear successful visualization. Their errors must not
be reported as representative accuracy; use the complete held-out evaluation.
`index.json` records original test IDs, hashes, model attribution, and object counts.

Python reproduction without the full dataset:

```bash
python -m pipeline.inference web_demo/samples/single_red.png \
  --camera web_demo/samples/single_red.camera.json --height-m 0.700 \
  --output reports/sample_red.png
```

## Attribution

Rendered images are distributed conservatively under **CC-BY-SA-3.0**.
The original traffic-cone3D model and textures are by **Mugicha3G**:
https://mugicha3.booth.pm/items/4001203

Source license: https://creativecommons.org/licenses/by-sa/3.0/

Changes: uniformly rescaled source mesh to0.700m height, generated procedural
backgrounds and lighting, varied materials and applied image degradation. No
third-party photographic backgrounds are used in these samples. The surrounding
demo source code remains under the repository's GPL-3.0-only code license.
