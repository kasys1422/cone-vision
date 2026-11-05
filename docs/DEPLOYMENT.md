# GitHub Pages / PWA

`main` root is the Pages source. `index.html` redirects to `web_demo/`.

1. Settings → Pages → Deploy from a branch → main / (root).
2. Verify https://kasys1422.github.io/cone-vision/ after deployment.
3. Open the site over HTTPS on a mobile device, grant camera access, shoot a photo, scan, then select a cone.
4. In settings, enter measured cone height/base width and camera intrinsics (`K` in a JSON file) for meaningful metric distances. A user-entered HFOV produces a rough estimate only.

`web_demo/sw.js` caches the static application, v7 ONNX models and bundled OpenCV.js. ONNX Runtime Web is loaded from jsDelivr, so the first install requires a network connection. Test offline support only after the models have been cached. Do not assume every browser or WebGPU implementation is validated.

This version is experimental and may produce false-positive cone detections and incorrect distances. It is not suitable for safety-critical measurement.
