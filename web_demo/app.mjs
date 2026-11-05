import {loadPackedScriptURL} from './packed_loader.mjs';
import {ConeVision} from './inference.mjs';
import {loadOpenCV, scaleGeometry} from './pnp.mjs';

const $ = sel => document.querySelector(sel);
const screens = {
  home: $('#homeScreen'),
  capture: $('#captureScreen'),
  review: $('#reviewScreen'),
  result: $('#resultScreen')
};
const reviewCanvas = $('#reviewCanvas');
const resultCanvas = $('#resultCanvas');
const reviewCtx = reviewCanvas.getContext('2d');
const resultCtx = resultCanvas.getContext('2d');
const cameraVideo = $('#cameraVideo');
const cameraCanvas = $('#cameraCanvas');

const palette = ['#d65dff', '#ffd34a', '#40c5ff', '#34e38d', '#ff4d8a'];

const state = {
  vision: null,
  cv: null,
  geometry: null,
  sourceBitmap: null,
  sourceImageData: null,
  lastResult: null,
  selectedIndex: null,
  cameraStream: null,
  facingMode: 'environment',
  installPrompt: null,
  cameraCalibration: null,
  modelLoading: false,
  scanning: false
};

const els = {
  installButton: $('#installButton'),
  imageInput: $('#imageInput'),
  openCameraButton: $('#openCameraButton'),
  closeCameraButton: $('#closeCameraButton'),
  switchCameraButton: $('#switchCameraButton'),
  captureButton: $('#captureButton'),
  retakeButton: $('#retakeButton'),
  scanButton: $('#scanButton'),
  rescanButton: $('#rescanButton'),
  saveResultButton: $('#saveResultButton'),
  thresholdInput: $('#thresholdInput'),
  thresholdLabel: $('#thresholdLabel'),
  useWebGPUInput: $('#useWebGPUInput'),
  refineRoiInput: $('#refineRoiInput'),
  coneHeightInput: $('#coneHeightInput'),
  baseWidthInput: $('#baseWidthInput'),
  fovInput: $('#fovInput'),
  cameraJsonInput: $('#cameraJsonInput'),
  loadingOverlay: $('#loadingOverlay'),
  coneList: $('#coneList'),
  objectCountBadge: $('#objectCountBadge'),
  detailPanel: $('#detailPanel'),
  resultSubtitle: $('#resultSubtitle'),
  distanceValue: $('#distanceValue'),
  depthValue: $('#depthValue'),
  confidenceValue: $('#confidenceValue'),
  modeValue: $('#modeValue'),
  detailNote: $('#detailNote')
};

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

function showScreen(name) {
  Object.entries(screens).forEach(([key, el]) => el.classList.toggle('active', key === name));
  if (name !== 'capture') stopCamera();
}

function setLoading(loading, label = 'スキャン中…') {
  state.scanning = loading;
  els.loadingOverlay.classList.toggle('hidden', !loading);
  els.loadingOverlay.innerHTML = loading ? `<span class="spinner"></span><span>${label}</span>` : '';
  els.scanButton.disabled = loading;
  els.rescanButton.disabled = loading;
}

function setModelLoading(loading) {
  state.modelLoading = loading;
  els.scanButton.disabled = loading || !state.sourceImageData;
}

function formatMeters(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} m` : '—';
}

function confidenceLabel(conf) {
  const score = Math.round(conf * 100);
  if (score >= 85) return `高 (${score}%)`;
  if (score >= 60) return `中 (${score}%)`;
  return `低 (${score}%)`;
}

function createApproxK(width, height, hfovDeg) {
  const hfov = (hfovDeg * Math.PI) / 180;
  const fx = (width / 2) / Math.tan(hfov / 2);
  const fy = fx;
  const cx = width / 2;
  const cy = height / 2;
  return [[fx, 0, cx], [0, fy, cy], [0, 0, 1]];
}

function activeGeometry() {
  const h = Number(els.coneHeightInput.value);
  const w = Number(els.baseWidthInput.value);
  return scaleGeometry(state.geometry, h, w);
}

function activeCameraModel(width, height) {
  if (state.cameraCalibration?.K) return {K: state.cameraCalibration.K, distortion: state.cameraCalibration.distortion ?? [], mode: '校正済み'};
  const hfov = Number(els.fovInput.value || 60);
  return {K: createApproxK(width, height, hfov), distortion: [], mode: '概算'};
}

async function loadModels() {
  setModelLoading(true);
  try {
    if (!state.geometry) {
      const response = await fetch('../assets/cone_geometry.json');
      if (!response.ok) throw new Error('cone_geometry.json を読み込めません');
      state.geometry = await response.json();
      els.coneHeightInput.value = state.geometry.height_m.toFixed(3);
      els.baseWidthInput.value = state.geometry.base_width_m.toFixed(3);
    }
    if (!state.cv) {
      const opencvURL = await loadPackedScriptURL('../.payload/opencv', 5);
      try { ({cv: state.cv} = await loadOpenCV(opencvURL)); }
      finally { URL.revokeObjectURL(opencvURL); }
    }
    if (state.vision) {
      await Promise.allSettled([state.vision.detector.release(), state.vision.keypoints.release()]);
      state.vision = null;
    }
    state.vision = await ConeVision.create(ort, {preferWebGPU: els.useWebGPUInput.checked, detectorProfile: 'retrain_v7'});
  } finally {
    setModelLoading(false);
  }
}

async function ensureSourceImageData() {
  if (state.sourceImageData) return state.sourceImageData;
  if (!state.sourceBitmap) throw new Error('画像がありません');
  const cvs = document.createElement('canvas');
  cvs.width = state.sourceBitmap.width;
  cvs.height = state.sourceBitmap.height;
  const ctx = cvs.getContext('2d');
  ctx.drawImage(state.sourceBitmap, 0, 0);
  state.sourceImageData = ctx.getImageData(0, 0, cvs.width, cvs.height);
  return state.sourceImageData;
}

function drawBitmapToCanvas(bitmap, canvas, ctx) {
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0);
}

function renderResultCanvas() {
  if (!state.sourceBitmap) return;
  drawBitmapToCanvas(state.sourceBitmap, resultCanvas, resultCtx);
  const result = state.lastResult;
  if (!result) return;
  const factor = Math.max(1, resultCanvas.width / 640);
  resultCtx.lineWidth = 2 * factor;
  resultCtx.font = `${13 * factor}px system-ui`;
  result.objects.forEach((obj, index) => {
    const selected = index === state.selectedIndex;
    const [x1, y1, x2, y2] = obj.bbox_xyxy;
    resultCtx.strokeStyle = selected ? '#97ff3a' : '#32d97d';
    resultCtx.shadowColor = selected ? 'rgba(151,255,58,0.55)' : 'transparent';
    resultCtx.shadowBlur = selected ? 20 : 0;
    resultCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    resultCtx.shadowBlur = 0;
    const label = `cone ${(obj.confidence * 100).toFixed(0)}%`;
    const yy = Math.max(18 * factor, y1 - 8 * factor);
    const textWidth = resultCtx.measureText(label).width + 8 * factor;
    resultCtx.fillStyle = 'rgba(22,24,28,.75)';
    resultCtx.fillRect(x1, yy - 14 * factor, textWidth, 18 * factor);
    resultCtx.fillStyle = '#7cf2c2';
    resultCtx.fillText(label, x1 + 4 * factor, yy);
    obj.keypoints.points.forEach(([x, y], i) => {
      resultCtx.beginPath();
      resultCtx.arc(x, y, 4.5 * factor, 0, Math.PI * 2);
      resultCtx.fillStyle = palette[i];
      resultCtx.strokeStyle = palette[i];
      if (obj.keypoints.visibility[i] === 2) resultCtx.fill(); else resultCtx.stroke();
    });
  });
}

function selectCone(index) {
  state.selectedIndex = index;
  renderResultCanvas();
  renderConeList();
  const obj = state.lastResult?.objects?.[index];
  if (!obj) {
    els.detailPanel.classList.add('hidden');
    els.resultSubtitle.textContent = 'コーンを選択してください';
    return;
  }
  els.detailPanel.classList.remove('hidden');
  els.resultSubtitle.textContent = `${index + 1}番目のコーンを選択中`;
  const pose = obj.pose;
  els.distanceValue.textContent = pose ? formatMeters(pose.distance_m) : '—';
  els.depthValue.textContent = pose ? formatMeters(pose.depth_m) : '—';
  els.confidenceValue.textContent = confidenceLabel(obj.confidence);
  els.modeValue.textContent = state.lastResult?.pose_mode ?? '—';
  const notes = [];
  if (state.lastResult?.pose_mode === '概算') notes.push('未校正の概算距離です。');
  if (pose && !pose.accepted) notes.push('PnPの品質判定では保留ですが、参考値として表示しています。');
  if (!pose) notes.push('カメラ設定が不足しているため距離を表示できません。');
  notes.push(`BBox信頼度 ${(obj.confidence * 100).toFixed(0)}%`);
  els.detailNote.textContent = notes.join(' ');
}


function selectConeByCanvasEvent(event) {
  if (!state.lastResult?.objects?.length) return;
  const rect = resultCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (resultCanvas.width / rect.width);
  const y = (event.clientY - rect.top) * (resultCanvas.height / rect.height);
  let best = -1;
  let bestArea = Infinity;
  state.lastResult.objects.forEach((obj, index) => {
    const [x1, y1, x2, y2] = obj.bbox_xyxy;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      const area = (x2 - x1) * (y2 - y1);
      if (area < bestArea) {
        bestArea = area;
        best = index;
      }
    }
  });
  if (best >= 0) selectCone(best);
}

function renderConeList() {
  const objects = state.lastResult?.objects ?? [];
  els.objectCountBadge.textContent = String(objects.length);
  if (!objects.length) {
    els.coneList.className = 'cone-list empty';
    els.coneList.textContent = 'コーンが見つかりませんでした';
    els.detailPanel.classList.add('hidden');
    return;
  }
  els.coneList.className = 'cone-list';
  els.coneList.innerHTML = '';
  objects.forEach((obj, index) => {
    const item = document.createElement('button');
    item.type = 'button';
    item.className = `cone-item ${index === state.selectedIndex ? 'selected' : ''}`;
    const pose = obj.pose;
    item.innerHTML = `
      <div class="cone-rank">${index + 1}</div>
      <div class="cone-meta">
        <strong>${pose ? formatMeters(pose.distance_m) : '距離なし'}</strong>
        <small>信頼度 ${(obj.confidence * 100).toFixed(0)}% / BBox ${(obj.bbox_xyxy[2] - obj.bbox_xyxy[0]).toFixed(0)}×${(obj.bbox_xyxy[3] - obj.bbox_xyxy[1]).toFixed(0)} px</small>
      </div>
      <span class="conf-chip">${confidenceLabel(obj.confidence)}</span>`;
    item.addEventListener('click', () => selectCone(index));
    els.coneList.appendChild(item);
  });
}

async function runScan() {
  if (!state.vision) await loadModels();
  const imageData = await ensureSourceImageData();
  setLoading(true);
  try {
    const points3d = activeGeometry();
    const camera = activeCameraModel(imageData.width, imageData.height);
    state.lastResult = await state.vision.predict(imageData, {
      threshold: Number(els.thresholdInput.value),
      keypointVariant: els.refineRoiInput.checked ? 'roi_refine' : 'baseline',
      points3d,
      K: camera.K,
      distortion: camera.distortion,
      cv: state.cv
    });
    state.lastResult.pose_mode = camera.mode;
    showScreen('result');
    state.selectedIndex = state.lastResult.objects.length ? 0 : null;
    renderResultCanvas();
    renderConeList();
    if (state.selectedIndex !== null) selectCone(state.selectedIndex); else selectCone(null);
  } catch (error) {
    alert(`スキャンに失敗しました: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function setSourceFromBitmap(bitmap) {
  if (state.sourceBitmap?.close) {
    try { state.sourceBitmap.close(); } catch {}
  }
  state.sourceBitmap = bitmap;
  state.sourceImageData = null;
  state.lastResult = null;
  state.selectedIndex = null;
  drawBitmapToCanvas(bitmap, reviewCanvas, reviewCtx);
  showScreen('review');
  els.scanButton.disabled = state.modelLoading;
}

async function startCamera() {
  stopCamera();
  const constraints = {
    audio: false,
    video: {
      facingMode: {ideal: state.facingMode},
      width: {ideal: 1280},
      height: {ideal: 720}
    }
  };
  state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
  cameraVideo.srcObject = state.cameraStream;
  await cameraVideo.play();
  showScreen('capture');
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(track => track.stop());
    state.cameraStream = null;
  }
  cameraVideo.srcObject = null;
}

async function captureFrame() {
  if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) return;
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
  const bitmap = await createImageBitmap(cameraCanvas);
  await setSourceFromBitmap(bitmap);
}

async function handleFileInput(file) {
  const bitmap = await createImageBitmap(file);
  await setSourceFromBitmap(bitmap);
}

function registerServiceWorker() {
  if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
}

function setupInstallPrompt() {
  window.addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    state.installPrompt = event;
    els.installButton.classList.remove('hidden');
  });
  els.installButton.addEventListener('click', async () => {
    if (!state.installPrompt) return;
    state.installPrompt.prompt();
    await state.installPrompt.userChoice;
    state.installPrompt = null;
    els.installButton.classList.add('hidden');
  });
}

function saveResultJson() {
  if (!state.lastResult) return;
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cone_vision_result.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function bindEvents() {
  document.querySelectorAll('[data-back]').forEach(btn => btn.addEventListener('click', () => showScreen(btn.dataset.back)));
  els.thresholdInput.addEventListener('input', () => { els.thresholdLabel.value = Number(els.thresholdInput.value).toFixed(2); });
  els.openCameraButton.addEventListener('click', async () => {
    try { await startCamera(); } catch (error) { alert(`カメラを開始できません: ${error.message}`); }
  });
  els.closeCameraButton.addEventListener('click', () => showScreen('home'));
  els.switchCameraButton.addEventListener('click', async () => {
    state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
    try { await startCamera(); } catch (error) { alert(`カメラ切替に失敗しました: ${error.message}`); }
  });
  els.captureButton.addEventListener('click', captureFrame);
  els.retakeButton.addEventListener('click', startCamera);
  els.scanButton.addEventListener('click', runScan);
  els.rescanButton.addEventListener('click', runScan);
  els.imageInput.addEventListener('change', async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    try { await handleFileInput(file); } catch (error) { alert(`画像を読み込めません: ${error.message}`); }
    event.target.value = '';
  });
  els.cameraJsonInput.addEventListener('change', async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      state.cameraCalibration = JSON.parse(await file.text());
      if (!Array.isArray(state.cameraCalibration.K)) throw new Error('K がありません');
    } catch (error) {
      state.cameraCalibration = null;
      alert(`校正JSONを読めません: ${error.message}`);
    }
  });
  els.useWebGPUInput.addEventListener('change', loadModels);
  resultCanvas.addEventListener('click', selectConeByCanvasEvent);
  els.saveResultButton.addEventListener('click', saveResultJson);
}

async function bootstrap() {
  bindEvents();
  setupInstallPrompt();
  registerServiceWorker();
  try {
    await loadModels();
  } catch (error) {
    alert(`モデルの初期化に失敗しました: ${error.message}`);
  }
}

bootstrap();
