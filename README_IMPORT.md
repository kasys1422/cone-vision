# Cone Vision PWA deployment

The camera-style PWA is included in `web_demo/`, with the v7 ONNX models in `models/retrain_v7/`.
Host the repository with an HTTP server (`python -m http.server 8877`) or GitHub Pages. The app opens from `/` and redirects to `/web_demo/`.

Features: camera capture, gallery import, scan, multi-cone selection, 2D keypoints, experimental range display, PWA manifest and Service Worker.

Caveats: model quality on real photographs has not been validated; calibration is required for precise metric ranging. The 60-degree HFOV default is a user-configurable approximation only. The app uses ONNX Runtime Web from jsDelivr, so first load requires network access.
