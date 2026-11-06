import {loadPackedScriptURL} from './packed_loader.mjs';
import {convertFovSpan, readExifFov} from './exif_fov.mjs';
import {deleteModels, depthProfiles, mib, modelStorage, readModel} from './model_store.mjs';
import {inferCloud, measuredCones} from './depth_cloud.mjs';
import {openViewer} from './cloud_view.mjs';
import {encodeLas} from './las.mjs';
import {ConeVision, DETECTOR_PROFILES, sameCone} from './inference.mjs';
import {loadOpenCV, scaleGeometry} from './pnp.mjs';

const $ = sel => document.querySelector(sel);
const screens = {
  home: $('#homeScreen'),
  capture: $('#captureScreen'),
  review: $('#reviewScreen'),
  result: $('#resultScreen'),
  cloud: $('#cloudScreen')
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
  removedCones: [],
  cloud: null,
  viewer: null,
  selectedIndex: null,
  regionSelection: null,
  cameraStream: null,
  facingMode: 'environment',
  installPrompt: null,
  cameraCalibration: null,
  scanning: false,
  imageSequence: 0,
  exifFov: null,
  manualLongEdgeFov: 60,
  fovSource: 'manual'
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
  detectorProfile: $('#detectorProfile'),
  thresholdLabel: $('#thresholdLabel'),
  useWebGPUInput: $('#useWebGPUInput'),
  refineRoiInput: $('#refineRoiInput'),
  coneHeightInput: $('#coneHeightInput'),
  baseWidthInput: $('#baseWidthInput'),
  fovInput: $('#fovInput'),
  reviewFovInput: $('#reviewFovInput'),
  useExifFovButton: $('#useExifFovButton'),
  cameraJsonInput: $('#cameraJsonInput'),
  dialog: $('#scanDialog'),
  coneList: $('#coneList'),
  objectCountBadge: $('#objectCountBadge'),
  detailPanel: $('#detailPanel'),
  resultSubtitle: $('#resultSubtitle'),
  distanceValue: $('#distanceValue'),
  depthValue: $('#depthValue'),
  confidenceValue: $('#confidenceValue'),
  selectedConeTitle: $('#selectedConeTitle')
};

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

function showScreen(name) {
  if (name !== 'cloud') {
    state.viewer?.close();
    state.viewer = null;
  }
  if (name !== 'result') setRegionMode(false);
  Object.entries(screens).forEach(([key, el]) => el.classList.toggle('active', key === name));
  if (name !== 'capture') stopCamera();
}

function setBusy(busy) {
  state.scanning = busy;
  els.scanButton.disabled = busy || !state.sourceBitmap;
  els.rescanButton.disabled = busy;
  if (busy) els.dialog.showModal();
  else els.dialog.close();
}

function work(title, detail = '', received, total) {
  if ($('#workTitle').textContent !== title) $('#workTitle').textContent = title;
  if ($('#workDetail').textContent !== detail) $('#workDetail').textContent = detail;
  const progress = $('#transferProgress');
  progress.classList.toggle('hidden', !total);
  if (total) { progress.max = total; progress.value = received; }
}

// Give the status a paint before the single-threaded WASM call occupies the UI thread.
const paint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

async function refreshStorage(prefix = '') {
  const profile = DETECTOR_PROFILES[els.detectorProfile.value];
  const info = await modelStorage(profile);
  if (profile !== DETECTOR_PROFILES[els.detectorProfile.value]) return;
  const hint = info.missing
    ? 'スキャン時に必要なモデルをダウンロードします（' + mib(info.missing) + ' MiB）。'
    : '保存済みのモデルでスキャンできます。';
  $('#modelStatus').textContent = prefix + (info.available ? '保存済み ' + mib(info.saved) + ' MiB。' : 'このブラウザでは保存できません。') + hint;
  $('#scanHint').textContent = state.vision ? 'モデル準備完了' : hint;
  $('#deleteModels').disabled = !(info.saved || state.vision);
}

async function releaseModels() {
  const vision = state.vision;
  state.vision = null;
  if (vision) await Promise.allSettled([vision.detector.release(), vision.keypoints.release()]);
}

function formatMeters(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)} m` : '—';
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
  if (state.cameraCalibration?.K) {
    const c = state.cameraCalibration;
    const expectedWidth = c.image_width ?? c.image_size?.[0];
    const expectedHeight = c.image_height ?? c.image_size?.[1];
    if ((expectedWidth && expectedWidth !== width) || (expectedHeight && expectedHeight !== height)) {
      throw new Error('校正JSONの画像サイズが読み込み画像と一致しません');
    }
    return {K: c.K, distortion: c.distortion ?? [], mode: '校正済み'};
  }
  const hfov = Number(els.fovInput.value);
  if (!(hfov > 4 && hfov < 175)) throw new Error('水平画角を4°～175°の範囲で指定してください');
  const isExif = state.fovSource === 'exif' && state.exifFov?.status === 'ok';
  // Calculated from oriented, decoded pixels; no EXIF GPS or private fields are retained.
  return {K: createApproxK(width, height, hfov), distortion: [], mode: isExif ? 'EXIF概算' : '手動概算'};
}

function horizontalSpanRatio() {
  const bitmap = state.sourceBitmap;
  return bitmap ? bitmap.width / Math.max(bitmap.width, bitmap.height) : 1;
}

function applyExifFov(result) {
  state.exifFov = result;
  const auto = result?.status === 'ok';
  state.fovSource = auto ? 'exif' : 'manual';
  const horizontal = auto ? result.fovDeg : convertFovSpan(state.manualLongEdgeFov, horizontalSpanRatio());
  els.fovInput.value = horizontal.toFixed(2);
  els.reviewFovInput.value = els.fovInput.value;
  els.useExifFovButton.classList.add('hidden');
}

function markManualFov(fromReview = false) {
  if (fromReview) els.fovInput.value = els.reviewFovInput.value;
  else els.reviewFovInput.value = els.fovInput.value;
  state.fovSource = 'manual';
  const degrees = Number(els.fovInput.value);
  if (degrees > 4 && degrees < 175) {
    state.manualLongEdgeFov = convertFovSpan(degrees, 1 / horizontalSpanRatio());
  }
  els.useExifFovButton.classList.toggle('hidden', state.exifFov?.status !== 'ok');
}

async function loadModels() {
  if (!state.geometry) {
    const response = await fetch('../assets/cone_geometry.json');
    if (!response.ok) throw new Error('cone_geometry.json を読み込めません');
    state.geometry = await response.json();
  }
  if (!state.cv) {
    const opencvURL = await loadPackedScriptURL('../.payload/opencv', 5);
    try { ({cv: state.cv} = await loadOpenCV(opencvURL)); }
    finally { URL.revokeObjectURL(opencvURL); }
  }
  const profile = DETECTOR_PROFILES[els.detectorProfile.value];
  const detectorBytes = await readWithProgress(profile.model, '検出モデル');
  const keypointBytes = await readWithProgress(profile.keypoints_model, '特徴点モデル');
  work('モデルを初期化中', '端末の推論エンジンを準備しています。');
  await paint();
  state.vision = await ConeVision.create(ort, {detectorProfile: els.detectorProfile.value,
    detectorBytes, keypointBytes, preferWebGPU: els.useWebGPUInput.checked});
}

function readWithProgress(path, label) {
  return readModel(path, (stage, received, total) => {
    if (stage === 'saved') work('保存済みモデルを読み込み中', label);
    else work('モデルをダウンロード中', label + '　' + mib(received) + ' / ' + mib(total) + ' MiB', received, total);
  });
}

// Previous versions stored the default on startup. Only remember explicit choices now.
const depthPreferenceKey = 'cone-depth-model-choice';
function chooseDepth(profile, remember = false) {
  for (const select of [$('#depthProfile'), $('#cloudProfile')]) select.value = profile;
  if (remember) {
    try { localStorage.setItem(depthPreferenceKey, profile); } catch {}
  }
}

function setDepthView(panelId) {
  for (const button of document.querySelectorAll('.depth-views button')) {
    const panel = document.getElementById(button.getAttribute('aria-controls'));
    const selected = panel.id === panelId;
    button.setAttribute('aria-pressed', String(selected));
    panel.classList.toggle('hidden', !selected);
  }
  for (const id of ['#gridControl', '#resetCloud']) $(id).classList.toggle('hidden', panelId !== 'pointPanel');
}

function renderDepthMap({depthMap: {metres, width, height}, far}) {
  const canvas = $('#depthCanvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  // Sample the same linear colour scale shown in the legend.
  const stops = [[0,'#fff099'], [.125,'#f9a247'], [.25,'#c74766'], [.5,'#64439a'], [1,'#233b72']];
  $('#depthScale').style.backgroundImage = 'linear-gradient(to right, ' +
    stops.map(([t, color]) => color + ' ' + t*100 + '%').join(', ') + ')';
  const ramp = document.createElement('canvas');
  ramp.width = 256; ramp.height = 1;
  const palette = ramp.getContext('2d');
  const gradient = palette.createLinearGradient(.5, 0, 255.5, 0);
  for (const [t, color] of stops) gradient.addColorStop(t, color);
  palette.fillStyle = gradient;
  palette.fillRect(0, 0, 256, 1);
  const colors = palette.getImageData(0, 0, 256, 1).data;
  const image = ctx.createImageData(width, height);
  for (let i = 0; i < metres.length; i++) {
    if (!metres[i]) continue;
    const k = Math.min(255, Math.round(metres[i]/far*255))*4;
    image.data[i*4] = colors[k]; image.data[i*4+1] = colors[k+1];
    image.data[i*4+2] = colors[k+2]; image.data[i*4+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  const label = n => n.toLocaleString('ja-JP', {maximumFractionDigits: 1}) + ' m';
  $('#midDepth').textContent = label(far/2);
  $('#farDepth').textContent = '遠 · ' + label(far);
}

function setCloudViewStatus(message = '') {
  $('#cloudCanvas').classList.toggle('hidden', Boolean(message));
  $('#cloudViewStatus').textContent = message;
  $('#cloudViewStatus').classList.toggle('hidden', !message);
  $('#resetCloud').disabled = $('#cloudGrid').disabled = Boolean(message);
}

async function showCloud(profile, remember = false) {
  if (state.scanning || measuredCones(state.lastResult?.objects ?? []).length < 2) return;
  setBusy(true);
  try {
    let cloud = state.cloud;
    if (cloud?.profile !== profile) {
      await releaseModels();
      const spec = depthProfiles[profile];
      const bytes = await readWithProgress(spec.url, spec.name);
      work('奥行きを推論中', spec.name);
      await paint();
      cloud = await inferCloud(ort, bytes, profile, state.sourceBitmap, state.lastResult, state.cv, els.useWebGPUInput.checked);
      cloud.profile = profile;
    }
    state.viewer?.close();
    state.viewer = null;
    if (!screens.cloud.classList.contains('active')) setDepthView('pointPanel');
    if (cloud !== state.cloud) {
      renderDepthMap(cloud);
      const canvas = $('#colorCanvas');
      canvas.width = cloud.colorImage.width;
      canvas.height = cloud.colorImage.height;
      canvas.getContext('2d').putImageData(cloud.colorImage, 0, 0);
    }
    showScreen('cloud');
    setCloudViewStatus();
    try { state.viewer = openViewer($('#cloudCanvas'), cloud, $('#cloudGrid')); }
    catch { setCloudViewStatus('3D描画を利用できません。LASを保存できます。'); }
    state.cloud = cloud;
    chooseDepth(profile, remember);
    $('#cloudInfo').textContent = 'コーン' + cloud.anchors.length + '個で補正 · ' +
      (cloud.xyz.length/3).toLocaleString() + '点';
  } catch (error) {
    if (screens.cloud.classList.contains('active') && state.cloud) chooseDepth(state.cloud.profile);
    else showScreen('result');
    alert('3D表示に失敗しました：' + (error.message ?? error));
  } finally {
    setBusy(false);
    await refreshStorage();
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
    els.resultSubtitle.textContent = state.lastResult?.objects?.length ? 'コーンを選択してください' : els.coneList.textContent;
    return;
  }
  els.detailPanel.classList.remove('hidden');
  els.resultSubtitle.textContent = `コーン ${index + 1}を選択`;
  els.selectedConeTitle.textContent = `コーン ${index + 1}`;
  const pose = obj.pose;
  els.distanceValue.textContent = pose ? formatMeters(pose.distance_m) : '—';
  els.depthValue.textContent = pose ? formatMeters(pose.depth_m) : '—';
  els.confidenceValue.textContent = obj.confidence.toFixed(2);
}


function removeSelectedCone() {
  const objects = state.lastResult?.objects;
  const index = state.selectedIndex;
  if (state.scanning || !Number.isInteger(index) || !objects?.[index]) return;
  const [object] = objects.splice(index, 1);
  state.removedCones.push({index, object});
  state.cloud = null;
  setRegionMode(false);
  selectCone(null);
  $('#undoRemoveCone').focus({preventScroll: true});
}

function undoRemoveCone() {
  if (state.scanning || !state.removedCones.length) return;
  const {index, object} = state.removedCones.pop();
  state.lastResult.objects.splice(index, 0, object);
  state.cloud = null;
  setRegionMode(false);
  selectCone(index);
  els.coneList.children[index].focus({preventScroll: true});
}

function selectConeByCanvasEvent(event) {
  if (state.scanning || state.regionSelection || !state.lastResult?.objects?.length) return;
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

function setRegionMode(active, message = '') {
  state.regionSelection = active ? {box: null, pointerId: null, start: null} : null;
  resultCanvas.classList.toggle('selecting-region', active);
  $('#chooseRegion').classList.toggle('hidden', active);
  $('#cancelRegion').classList.toggle('hidden', !active);
  $('#regionStatus').textContent = active ? 'コーンの周囲をドラッグし、離すと再検出します。' : message;
  drawRegionFrame();
}

function drawRegionFrame() {
  const box = state.regionSelection?.box;
  const frame = $('#regionFrame');
  frame.classList.toggle('hidden', !box);
  if (!box) return;
  const rect = resultCanvas.getBoundingClientRect();
  const sx = rect.width / resultCanvas.width, sy = rect.height / resultCanvas.height;
  Object.assign(frame.style, {
    left: (resultCanvas.offsetLeft + box[0]*sx)+'px', top: (resultCanvas.offsetTop + box[1]*sy)+'px',
    width: ((box[2]-box[0])*sx)+'px', height: ((box[3]-box[1])*sy)+'px'
  });
}

function regionPoint(event) {
  const rect = resultCanvas.getBoundingClientRect();
  const pixel = (p, span, size) => Math.round(Math.max(0, Math.min(1, p/span)) * size);
  return [pixel(event.clientX-rect.left, rect.width, resultCanvas.width),
          pixel(event.clientY-rect.top, rect.height, resultCanvas.height)];
}

function bindRegionSelection() {
  $('#chooseRegion').addEventListener('click', () => setRegionMode(true));
  $('#cancelRegion').addEventListener('click', () => setRegionMode(false));
  new ResizeObserver(drawRegionFrame).observe(resultCanvas);
  resultCanvas.addEventListener('pointerdown', event => {
    const selection = state.regionSelection;
    if (!selection || !event.isPrimary || event.button !== 0) return;
    selection.start = regionPoint(event);
    selection.pointerId = event.pointerId;
    selection.box = null;
    resultCanvas.setPointerCapture(event.pointerId);
    drawRegionFrame();
  });
  const update = event => {
    const selection = state.regionSelection;
    if (selection?.pointerId !== event.pointerId) return;
    const [x,y] = regionPoint(event), [a,b] = selection.start;
    selection.box = [Math.min(x,a), Math.min(y,b), Math.max(x,a), Math.max(y,b)];
    if (event.type === 'pointerup') {
      selection.pointerId = null;
      const rect = resultCanvas.getBoundingClientRect();
      if (Math.abs(x-a)*rect.width/resultCanvas.width < 6 || Math.abs(y-b)*rect.height/resultCanvas.height < 6) {
        selection.box = null;
      } else {
        runScan(selection.box);
      }
    }
    drawRegionFrame();
  };
  resultCanvas.addEventListener('pointermove', update);
  resultCanvas.addEventListener('pointerup', update);
  resultCanvas.addEventListener('lostpointercapture', event => {
    if (state.regionSelection?.pointerId === event.pointerId) setRegionMode(true);
  });
}

function renderConeList() {
  const objects = state.lastResult?.objects ?? [];
  $('#undoRemoveCone').classList.toggle('hidden', !state.removedCones.length);
  $('#openCloud').classList.toggle('hidden', measuredCones(objects).length < 2);
  els.objectCountBadge.textContent = String(objects.length);
  if (!objects.length) {
    els.coneList.className = 'cone-list empty';
    els.coneList.textContent = state.removedCones.length ? '残っているコーンはありません' : 'コーンが見つかりませんでした';
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
    item.setAttribute('aria-pressed', String(index === state.selectedIndex));
    item.setAttribute('aria-label', `コーン ${index + 1}、${pose ? formatMeters(pose.distance_m) : '距離なし'}`);
    item.innerHTML = `
      <div class="cone-rank">${index + 1}</div>
      <div class="cone-meta">
        <strong>${pose ? formatMeters(pose.distance_m) : '距離なし'}</strong>
      </div>
      <span class="selection-mark" aria-hidden="true">${index === state.selectedIndex ? '✓' : ''}</span>`;
    item.addEventListener('click', () => selectCone(index));
    els.coneList.appendChild(item);
  });
}

async function runScan(region = null) {
  if (state.scanning) return;
  setBusy(true);
  work('スキャンを準備中', '画像処理エンジンを準備しています。');
  try {
    if (!state.vision) await loadModels();
    work('画像を推論中');
    await paint();
    const imageData = await ensureSourceImageData();
    const points3d = activeGeometry();
    const camera = activeCameraModel(imageData.width, imageData.height);
    const result = await state.vision.predict(imageData, {
      region,
      threshold: Number(els.thresholdInput.value),
      keypointVariant: els.refineRoiInput.checked ? 'roi_refine' : 'baseline',
      points3d,
      K: camera.K,
      distortion: camera.distortion,
      cv: state.cv
    });
    if (region) {
      const objects = state.lastResult.objects;
      const previousCount = objects.length;
      let matchedIndex = null;
      for (const obj of result.objects) {
        if (state.removedCones.some(({object}) => sameCone(object.bbox_xyxy, obj.bbox_xyxy))) continue;
        const index = objects.findIndex(old => sameCone(old.bbox_xyxy, obj.bbox_xyxy));
        if (index < 0) objects.push(obj);
        else if (matchedIndex === null) matchedIndex = index;
      }
      const added = objects.length - previousCount;
      if (added) state.cloud = null;
      state.lastResult.time_ms += result.time_ms;
      setRegionMode(false, added ? `${added}個追加しました。` :
        matchedIndex !== null ? 'すでに検出済みです。' :
        result.objects.length ? '削除済みのコーンです。' : 'この範囲にコーンは見つかりませんでした。');
      selectCone(added ? previousCount : matchedIndex ?? state.selectedIndex);
      return;
    }
    state.cloud = null;
    state.lastResult = result;
    state.removedCones = [];
    result.pose_mode = camera.mode;
    result.camera_fov = {horizontal_fov_deg: Number(els.fovInput.value), source: state.fovSource, exif_status: state.exifFov?.status ?? 'none'};
    setRegionMode(false);
    showScreen('result');
    selectCone(result.objects.length ? 0 : null);
  } catch (error) {
    alert(`スキャンに失敗しました: ${error.message}`);
  } finally {
    setBusy(false);
    await refreshStorage();
  }
}

function setSourceFromBitmap(bitmap, source) {
  if (state.sourceBitmap?.close) {
    try { state.sourceBitmap.close(); } catch {}
  }
  state.sourceBitmap = bitmap;
  state.sourceImageData = null;
  state.cloud = null;
  state.lastResult = null;
  state.removedCones = [];
  state.selectedIndex = null;
  drawBitmapToCanvas(bitmap, reviewCanvas, reviewCtx);
  els.retakeButton.classList.toggle('hidden', source !== 'camera');
  showScreen('review');
  els.scanButton.disabled = false;
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
  const sequence = ++state.imageSequence;
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);
  const bitmap = await createImageBitmap(cameraCanvas);
  if (sequence !== state.imageSequence) { bitmap.close(); return; }
  setSourceFromBitmap(bitmap, 'camera');
  applyExifFov(null);
}

async function handleFileInput(file) {
  const sequence = ++state.imageSequence;
  const bitmap = await createImageBitmap(file);
  // Parse the *original* file before the canvas removes the metadata.
  const exif = await readExifFov(file, {width: bitmap.width, height: bitmap.height});
  if (sequence !== state.imageSequence) { bitmap.close(); return; }
  setSourceFromBitmap(bitmap, 'file');
  applyExifFov(exif);
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
  downloadFile(blob, 'cone_vision_result.json');
}

function downloadFile(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function bindEvents() {
  bindRegionSelection();
  for (const select of [$('#depthProfile'), $('#cloudProfile')]) {
    for (const [key, spec] of Object.entries(depthProfiles)) {
      select.add(new Option(spec.name + '（' + Math.round(spec.size / 1e6) + ' MB）', key));
    }
  }
  let preferred;
  try { preferred = localStorage.getItem(depthPreferenceKey); } catch {}
  chooseDepth(Object.hasOwn(depthProfiles, preferred) ? preferred : 'v2');
  $('#depthProfile').addEventListener('change', e => chooseDepth(e.target.value, true));
  $('#openCloud').addEventListener('click', () => showCloud($('#depthProfile').value));
  $('#cloudProfile').addEventListener('change', e => showCloud(e.target.value, true));
  for (const button of document.querySelectorAll('.depth-views button')) {
    button.addEventListener('click', () => setDepthView(button.getAttribute('aria-controls')));
  }
  $('#resetCloud').addEventListener('click', () => state.viewer?.reset());
  $('#cloudCanvas').addEventListener('webglcontextlost', event => {
    event.preventDefault();
    state.viewer?.close();
    state.viewer = null;
    setCloudViewStatus('3D描画が中断しました。LASを保存できます。');
  });
  $('#saveLas').addEventListener('click', () => {
    try {
      const bytes = encodeLas(state.cloud);
      downloadFile(new Blob([bytes], {type:'application/octet-stream'}), 'cone-vision-' + state.cloud.profile + '.las');
    } catch (error) { alert('LASを保存できませんでした：' + error.message); }
  });
  document.querySelectorAll('[data-back]').forEach(btn => btn.addEventListener('click', () => showScreen(btn.dataset.back)));
  els.thresholdInput.addEventListener('input', () => { els.thresholdLabel.value = Number(els.thresholdInput.value).toFixed(2); });
  els.fovInput.addEventListener('input', () => markManualFov(false));
  els.reviewFovInput.addEventListener('input', () => markManualFov(true));
  els.useExifFovButton.addEventListener('click', () => applyExifFov(state.exifFov));
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
  els.scanButton.addEventListener('click', () => runScan());
  els.rescanButton.addEventListener('click', () => runScan());
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
  const changeModel = async () => {
    await releaseModels();
    await refreshStorage();
  };
  els.useWebGPUInput.addEventListener('change', () => {
    state.cloud = null;
    return changeModel();
  });
  els.detectorProfile.addEventListener('change', changeModel);
  els.dialog.addEventListener('cancel', event => event.preventDefault());
  $('#deleteModels').addEventListener('click', async () => {
    setBusy(true);
    work('モデルを削除中', '端末に保存した全モデルを削除しています。');
    try {
      await releaseModels();
      await deleteModels();
      state.cloud = null;
      await refreshStorage('削除しました。');
    } catch (error) {
      alert('モデルを削除できませんでした：' + error.message);
    } finally {
      setBusy(false);
    }
  });
  resultCanvas.addEventListener('click', selectConeByCanvasEvent);
  $('#removeCone').addEventListener('click', removeSelectedCone);
  $('#undoRemoveCone').addEventListener('click', undoRemoveCone);
  els.saveResultButton.addEventListener('click', saveResultJson);
}

async function bootstrap() {
  bindEvents();
  setupInstallPrompt();
  registerServiceWorker();
  await refreshStorage();
}

bootstrap();
