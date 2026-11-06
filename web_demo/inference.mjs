import {loadPackedBinary} from './packed_loader.mjs';
/** GPL-3.0-only. Two independent ONNX sessions; no server inference.
 * Geometry/PnP reference implementation lives in ../geometry/pnp.py.
 * All output pixels refer to the original uploaded image, with no EXIF ambiguity
 * after browser image decoding. Calibrate K in this same decoded orientation.
 */
import {solveConePose, poseSettingsForRoi} from './pnp.mjs';

const sigmoid = x => 1 / (1 + Math.exp(-x));
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const roundEven = x => {
  const lo = Math.floor(x), fraction = x - lo;
  return fraction === 0.5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(x);
};

function pixel(data, width, height, x, y, c, fill = 114) {
  return x < 0 || x >= width || y < 0 || y >= height ? fill : data[(y * width + x) * 4 + c];
}

function bilinear(data, width, height, x, y, c, fill, quantize = false, clampEdge = false) {
  if (clampEdge) {
    x = clip(x, 0, width - 1);
    y = clip(y, 0, height - 1);
  }
  if (quantize) {
    // OpenCV warpAffine uses a 32-entry interpolation phase table.
    x = Math.round(x * 32) / 32;
    y = Math.round(y * 32) / 32;
  }
  const x0 = Math.floor(x), y0 = Math.floor(y), dx = x - x0, dy = y - y0;
  const value = (1 - dy) * ((1 - dx) * pixel(data, width, height, x0, y0, c, fill) +
      dx * pixel(data, width, height, x0 + 1, y0, c, fill)) +
    dy * ((1 - dx) * pixel(data, width, height, x0, y0 + 1, c, fill) +
      dx * pixel(data, width, height, x0 + 1, y0 + 1, c, fill));
  return Math.round(value);
}

// Torchvision SSDLite resizes both axes independently, in floating point.
export function ssdInput({data, width, height}, size) {
  const tensor = new Float32Array(3 * size * size);
  const sx = size / width, sy = size / height;
  for (let y = 0; y < size; y++) {
    const fy = clip((y + .5) / sy - .5, 0, height - 1);
    const y0 = Math.floor(fy), y1 = Math.min(y0 + 1, height - 1), dy = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = clip((x + .5) / sx - .5, 0, width - 1);
      const x0 = Math.floor(fx), x1 = Math.min(x0 + 1, width - 1), dx = fx - x0;
      for (let c = 0; c < 3; c++) {
        const top = (1-dx)*data[(y0*width+x0)*4+c] + dx*data[(y0*width+x1)*4+c];
        const bottom = (1-dx)*data[(y1*width+x0)*4+c] + dx*data[(y1*width+x1)*4+c];
        tensor[c*size*size+y*size+x] = ((1-dy)*top + dy*bottom) / 255;
      }
    }
  }
  return {tensor, transform: {sx, sy, left: 0, top: 0, originalWidth: width, originalHeight: height, size}};
}

export function detectorInput(image, size = 256, cv = null) {
  const {data, width, height} = image;
  const scale = Math.min(size / width, size / height);
  // Python round() uses ties-to-even; exact half-integer dimensions are handled.
  const nw = Math.max(1, roundEven(width * scale)), nh = Math.max(1, roundEven(height * scale));
  const left = Math.floor((size - nw) / 2), top = Math.floor((size - nh) / 2);
  const sx = nw / width, sy = nh / height;
  const tensor = new Float32Array(3 * size * size).fill(114 / 255);
  if (cv) {
    // Match the native evaluator's uint8 RGB OpenCV INTER_LINEAR operation.
    // Ideal floating-point bilinear interpolation can differ by one uint8 level;
    // a sensitive ROI model can amplify the resulting detector box difference.
    const rgba = new cv.Mat(height, width, cv.CV_8UC4), rgb = new cv.Mat(), resized = new cv.Mat();
    try {
      rgba.data.set(data);
      cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
      cv.resize(rgb, resized, new cv.Size(nw, nh), 0, 0, cv.INTER_LINEAR);
      for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) for (let c = 0; c < 3; c++) {
        tensor[c * size * size + (y + top) * size + x + left] = resized.data[(y * nw + x) * 3 + c] / 255;
      }
    } finally { rgba.delete(); rgb.delete(); resized.delete(); }
    return {tensor, transform: {sx, sy, left, top, originalWidth: width, originalHeight: height, size}};
  }
  for (let y = 0; y < nh; y++) for (let x = 0; x < nw; x++) {
    const srcX = (x + 0.5) / sx - 0.5, srcY = (y + 0.5) / sy - 0.5;
    for (let c = 0; c < 3; c++) {
      tensor[c * size * size + (y + top) * size + x + left] =
        bilinear(data, width, height, srcX, srcY, c, 114, false, true) / 255;
    }
  }
  return {tensor, transform: {sx, sy, left, top, originalWidth: width, originalHeight: height, size}};
}

export function iou(a, b) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  const area = q => Math.max(0, q[2] - q[0]) * Math.max(0, q[3] - q[1]);
  return inter / Math.max(area(a) + area(b) - inter, 1e-9);
}

export function nms(detections, threshold = 0.5, maxDetections = 100) {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const out = [];
  for (const candidate of sorted) {
    if (out.every(kept => iou(candidate.bbox_xyxy, kept.bbox_xyxy) <= threshold)) out.push(candidate);
    if (out.length >= maxDetections) break;
  }
  return out;
}

export function decodeDetector(heatmap, geometry, transform, threshold = 0.25, nmsIou = 0.5) {
  const h = 64, w = 64, plane = h * w;
  const proposals = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const idx = y * w + x, score = heatmap[idx];
    if (score < threshold) continue;
    let maximum = true;
    for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++) {
      for (let xx = Math.max(0, x - 1); xx <= Math.min(w - 1, x + 1); xx++) {
        if (heatmap[yy * w + xx] > score) maximum = false;
      }
    }
    if (!maximum) continue;
    const cx = (x + sigmoid(clip(geometry[idx], -40, 40))) * 4;
    const cy = (y + sigmoid(clip(geometry[plane + idx], -40, 40))) * 4;
    const bw = Math.exp(clip(geometry[2 * plane + idx], -2, 6)) * 4;
    const bh = Math.exp(clip(geometry[3 * plane + idx], -2, 6)) * 4;
    const {sx, sy, left, top, originalWidth, originalHeight} = transform;
    const box = [clip((cx - bw / 2 - left) / sx, 0, originalWidth),
      clip((cy - bh / 2 - top) / sy, 0, originalHeight),
      clip((cx + bw / 2 - left) / sx, 0, originalWidth),
      clip((cy + bh / 2 - top) / sy, 0, originalHeight)];
    proposals.push({bbox_xyxy: box, confidence: score, class_id: 0});
  }
  proposals.sort((a, b) => b.confidence - a.confidence);
  return nms(proposals.slice(0, 100).filter(p => p.bbox_xyxy[2] > p.bbox_xyxy[0] + 1 &&
    p.bbox_xyxy[3] > p.bbox_xyxy[1] + 1), nmsIou);
}

export function decodeYolox(predictions, transform, threshold = 0.025, nmsIou = 0.5, topK = 1000, minSide = 1) {
  const candidates = [];
  for (let i = 0; i < predictions.length; i += 6) {
    const confidence = predictions[i + 4] * predictions[i + 5];
    if (confidence >= threshold) candidates.push({i, confidence});
  }
  candidates.sort((a, b) => b.confidence - a.confidence);
  const {sx, sy, left, top, originalWidth, originalHeight} = transform;
  const decoded = [];
  for (const {i, confidence} of candidates.slice(0, topK)) {
    const cx = predictions[i], cy = predictions[i + 1], bw = predictions[i + 2], bh = predictions[i + 3];
    const box = [clip((cx - bw / 2 - left) / sx, 0, originalWidth),
      clip((cy - bh / 2 - top) / sy, 0, originalHeight),
      clip((cx + bw / 2 - left) / sx, 0, originalWidth),
      clip((cy + bh / 2 - top) / sy, 0, originalHeight)];
    if (box[2] > box[0] + minSide && box[3] > box[1] + minSide) decoded.push({bbox_xyxy: box, confidence, class_id: 0});
  }
  return nms(decoded, nmsIou);
}

export function roiInput(image, bbox, size = 96, margin = 1.2, cv = null) {
  const {data, width, height} = image;
  const side = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1], 2) * margin;
  if (!(side > 0)) throw new Error('ROI must have positive size');
  const x0 = (bbox[0] + bbox[2] - side) / 2, y0 = (bbox[1] + bbox[3] - side) / 2;
  const tensor = new Float32Array(3 * size * size);
  if (cv) {
    const scale = size / side;
    const rgba = new cv.Mat(height, width, cv.CV_8UC4), rgb = new cv.Mat(), crop = new cv.Mat();
    const affine = cv.matFromArray(2, 3, cv.CV_32F, [scale, 0, -scale*x0, 0, scale, -scale*y0]);
    try {
      rgba.data.set(data);
      cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
      cv.warpAffine(rgb, crop, affine, new cv.Size(size, size), cv.INTER_LINEAR,
        cv.BORDER_CONSTANT, new cv.Scalar(114, 114, 114, 114));
      for (let p = 0; p < size * size; p++) for (let c = 0; c < 3; c++) tensor[c*size*size+p] = crop.data[3*p+c]/255;
    } finally { rgba.delete(); rgb.delete(); crop.delete(); affine.delete(); }
    return {tensor, roi: {x0, y0, side, size}};
  }
  // Mirror OpenCV's float32 forward affine, double inverse and 10-bit fixed
  // coordinate accumulator before selecting the 5-bit interpolation phase.
  const scale = size / side, affineScale = Math.fround(scale);
  const inverseScale = 1 / affineScale;
  const inverseX = -Math.fround(-scale * x0) / affineScale;
  const inverseY = -Math.fround(-scale * y0) / affineScale;
  const fixedX0 = roundEven(inverseX * 1024) + 16;
  const fixedY0 = roundEven(inverseY * 1024) + 16;
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const sx = Math.floor((fixedX0 + roundEven(inverseScale * x * 1024)) / 32) / 32;
    const sy = Math.floor((fixedY0 + roundEven(inverseScale * y * 1024)) / 32) / 32;
    for (let c = 0; c < 3; c++) tensor[c * size * size + y * size + x] =
      bilinear(data, width, height, sx, sy, c, 114, false) / 255;
  }
  return {tensor, roi: {x0, y0, side, size}};
}

export function decodeKeypoints(coordinates, confidence, visibility, roi) {
  const points = [], roiPoints = [], visibilityProbs = [], labels = [];
  for (let i = 0; i < 5; i++) {
    const x = coordinates[2 * i], y = coordinates[2 * i + 1];
    points.push([roi.x0 + x * roi.side, roi.y0 + y * roi.side]);
    roiPoints.push([x * roi.size, y * roi.size]);
    const v = Array.from(visibility.slice(i * 3, i * 3 + 3));
    visibilityProbs.push(v);
    labels.push(v.indexOf(Math.max(...v)));
  }
  return {points, roi_points: roiPoints, confidence: Array.from(confidence),
    visibility_probs: visibilityProbs, visibility: labels, roi};
}

export const DETECTOR_PROFILES = {
  v12_default: {
    model: '../models/tracon_v12/yolox_v10.onnx',
    keypoints_model: '../models/tracon_v12/cone_keypoints.onnx',
    architecture_id: 'yolox_nano', size: 256, keypoint_size: 192,
    recommended_threshold: .5, target: '交通コーン'
  },
  v12_mixed: {
    model: '../models/tracon_v12/ssdlite_mixed.onnx',
    keypoints_model: '../models/tracon_v12/cone_keypoints.onnx',
    architecture_id: 'ssdlite320_mobilenet_v3_large', size: 320, keypoint_size: 192,
    recommended_threshold: .5, target: '交通コーン'
  },
  tracon_v12: {
    model: '../models/tracon_v12/cone_detector.onnx',
    keypoints_model: '../models/tracon_v12/cone_keypoints.onnx',
    architecture_id: 'ssdlite320_mobilenet_v3_large', size: 320, keypoint_size: 192,
    recommended_threshold: .5, target: '交通コーン'
  },
  retrain_v7: {model: '../models/retrain_v7/cone_detector.onnx', keypoints_model: '../models/retrain_v7/cone_keypoints.onnx',
    size: 256, recommended_threshold: .5, architecture_id: 'yolox_nano',
    target: 'Complete-label synthetic v6 retraining; experimental, real-world accuracy not validated'},
  real_v3: {model: '../models/real_v3/cone_detector.onnx', keypoints_model: '../models/real_v3/cone_keypoints.onnx',
    size: 256, recommended_threshold: .5, architecture_id: 'yolox_nano',
    target: 'Small-data real fine-tuning candidate; not production-validated'},
  current: {model: '../models/cone_detector.onnx', size: 256, recommended_threshold: .5,
    architecture_id: 'yolox_nano', target: 'Traffic cones; synthetic red, yellow and blue training'},
  coneslayer: {model: '../models/external/coneslayer/coneslayer-simplified.onnx', size: 416,
    recommended_threshold: .54, architecture_id: 'coneslayer_yolov7_tiny', target: 'Orange cones per upstream',
    upstream_commit: 'ce7400b9bebb634227b09f88a53e37a54fac2479',
    sha256: '67c8dd8922fa793a0577297c8936de1fc5decaebfb7c79c1f9a7aa1542f93faf'}
};

const CORNER_PHASES = [[0,1,2,3,4], [0,2,3,4,1], [0,3,4,1,2], [0,4,1,2,3]];

export function refinementBox(first) {
  const {x0, y0, side} = first.roi;
  const lo = [0, 1].map(c => Math.min(...first.points.map(p => p[c])));
  const hi = [0, 1].map(c => Math.max(...first.points.map(p => p[c])));
  const old = [x0 + side / 2, y0 + side / 2];
  const center = old.map((v, c) => clip((lo[c] + hi[c]) / 2, v - .15 * side, v + .15 * side));
  const extent = clip(Math.max(hi[0] - lo[0], hi[1] - lo[1], 4), .8 * side / 1.2, 1.6 * side / 1.2);
  return [center[0] - extent/2, center[1] - extent/2, center[0] + extent/2, center[1] + extent/2];
}

export function fuseRefinedKeypoints(first, second) {
  let best = CORNER_PHASES[0], bestCost = Infinity;
  for (const perm of CORNER_PHASES) {
    let total = 0, weight = 0;
    first.points.forEach((p, i) => {
      const q = second.points[perm[i]], w = Math.max(.05, Math.min(first.confidence[i], second.confidence[perm[i]]));
      total += Math.hypot(p[0] - q[0], p[1] - q[1]) * w; weight += w;
    });
    if (total / weight < bestCost) { bestCost = total / weight; best = perm; }
  }
  const points = first.points.map((p, i) => p.map((v, c) => (v + second.points[best[i]][c]) / 2));
  const visibilityProbs = first.visibility_probs.map((p, i) => p.map((v, c) => (v + second.visibility_probs[best[i]][c]) / 2));
  return {...first, points,
    roi_points: points.map(p => [(p[0] - first.roi.x0) / first.roi.side * first.roi.size,
                                 (p[1] - first.roi.y0) / first.roi.side * first.roi.size]),
    confidence: first.confidence.map((v, i) => (v + second.confidence[best[i]]) / 2),
    visibility_probs: visibilityProbs,
    visibility: visibilityProbs.map(v => v.indexOf(Math.max(...v))),
    variant: 'roi_refine', crop_passes: 2, crop_rois: [first.roi, second.roi],
    cross_crop_std_px: first.points.map((p, i) => (Math.abs(p[0]-second.points[best[i]][0]) +
      Math.abs(p[1]-second.points[best[i]][1])) / 4),
    corner_identity: 'cyclic representative; phase unknown modulo 4', confidence_calibrated: false};
}

export class ConeVision {
  constructor(ort, detector, keypoints, provider, spec = {}) {
    this.ort = ort; this.detector = detector; this.keypoints = keypoints; this.provider = provider;
    this.spec = spec; this.recommendedThreshold = spec.recommended_threshold ?? 0.25;
  }
  static async create(ort, {detectorProfile = 'current', detectorURL = null,
    keypointsURL = null, preferWebGPU = true} = {}) {
    if (!Object.hasOwn(DETECTOR_PROFILES, detectorProfile)) throw new Error(`Unknown detector profile: ${detectorProfile}`);
    const profile = DETECTOR_PROFILES[detectorProfile];
    detectorURL ??= profile.model;
    keypointsURL ??= profile.keypoints_model ?? '../models/cone_keypoints.onnx';
    const choices = preferWebGPU && globalThis.navigator?.gpu ? ['webgpu', 'wasm'] : ['wasm'];
    let spec = {...profile, profile: detectorProfile, keypoints_model: keypointsURL};
    if (['current','real_v3','retrain_v7'].includes(detectorProfile) && typeof detectorURL === 'string') {
      try {
        const url = detectorURL.replace(/[^/]*$/, 'cone_detector.interface.json');
        const response = await fetch(url);
        if (response.ok) spec = {...spec, ...await response.json()};
      } catch { /* Explicit defaults keep the standalone interface usable. */ }
    }
    let lastError;
    const detectorSource = detectorProfile === 'retrain_v7'
      ? await loadPackedBinary('../.payload/detector', 5) : detectorURL;
    for (const provider of choices) {
      let detector;
      try {
        detector = await ort.InferenceSession.create(detectorSource, {executionProviders: [provider]});
        const keypoints = await ort.InferenceSession.create(keypointsURL, {executionProviders: [provider]});
        return new ConeVision(ort, detector, keypoints, provider, spec);
      } catch (err) {
        lastError = err;
        if (detector) await detector.release();
      }
    }
    throw lastError;
  }
  async predict(imageData, {threshold = this.recommendedThreshold, nmsIou = 0.5, points3d = null,
                           K = null, distortion = [], cv = null, keypointVariant = 'baseline'} = {}) {
    if (!['baseline', 'roi_refine'].includes(keypointVariant)) throw new Error(`Unsupported keypoint variant: ${keypointVariant}`);
    const started = performance.now();
    const size = this.spec.input?.shape?.[2] ?? this.spec.size ?? 256;
    const ssd = this.spec.architecture_id === 'ssdlite320_mobilenet_v3_large';
    const {tensor, transform} = ssd ? ssdInput(imageData, size) : detectorInput(imageData, size, cv);
    const feeds = {images: new this.ort.Tensor('float32', tensor, [1, 3, size, size])};
    const raw = this.spec.profile === 'coneslayer' ? await this.detector.run(feeds, ['output']) : await this.detector.run(feeds);
    const decoded = raw.predictions ?? raw.output;
    const detections = decoded ? decodeYolox(decoded.data, transform, threshold, nmsIou, ssd ? 300 : 1000, ssd ? 0 : 1) :
      decodeDetector(raw.heatmap.data, raw.geometry.data, transform, threshold, nmsIou);
    const objects = [];
    const inferKeypoints = async bbox => {
      const {tensor: roiTensor, roi} = roiInput(imageData, bbox, this.spec.keypoint_size ?? 96, 1.2, cv);
      const result = await this.keypoints.run({images: new this.ort.Tensor('float32', roiTensor, [1, 3, roi.size, roi.size])});
      return decodeKeypoints(result.coordinates.data, result.confidence.data, result.visibility.data, roi);
    };
    for (const detection of detections) {
      let keypoints = await inferKeypoints(detection.bbox_xyxy);
      if (keypointVariant === 'roi_refine') keypoints = fuseRefinedKeypoints(keypoints, await inferKeypoints(refinementBox(keypoints)));
      else keypoints = {...keypoints, variant: 'baseline', crop_passes: 1};
      const pose = cv && K && points3d ? solveConePose(cv, points3d, keypoints.points, K,
        keypoints.confidence, keypoints.visibility_probs, distortion, poseSettingsForRoi(keypoints.roi)) : null;
      objects.push({...detection, keypoints, points3d_m: points3d, pose,
        pnp_input: {points2d: keypoints.points, points3d_m: points3d, K,
          confidence: keypoints.confidence, visibility_probs: keypoints.visibility_probs,
          allowed_corner_permutations: [[0, 1, 2, 3, 4], [0, 2, 3, 4, 1], [0, 3, 4, 1, 2], [0, 4, 1, 2, 3]]}});
    }
    return {schema_version: 1, image_size: [imageData.width, imageData.height], objects, K, distortion,
      detector_profile: this.spec.profile ?? 'current', keypoint_model: this.spec.keypoints_model, keypoint_variant: keypointVariant,
      detection_threshold: threshold, detector_target: this.spec.target ?? null,
      detector_preprocessing: ssd ? 'Torchvision RGB float bilinear stretch' : cv ? 'OpenCV INTER_LINEAR uint8 RGB' : 'JavaScript bilinear; native OpenCV rounding may differ',
      roi_preprocessing: cv ? 'OpenCV warpAffine float32 affine, INTER_LINEAR, border114' : 'JavaScript fixed-point approximation',
      provider: this.provider, time_ms: performance.now() - started,
      notes: ['Corner phase is ambiguous modulo 90 degrees.',
        cv && K && points3d ? 'Metric pose uses OpenCV.js unweighted refinement with explicit C4 ambiguity and conditioning checks.' :
        'PnP-ready points exported; metric pose requires an OpenCV runtime, calibrated K and known physical anchor coordinates.',
        'Confidence is an uncalibrated model score, not an accuracy probability.']};
  }
}
