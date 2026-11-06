export const depthProfiles = {
  v2: {family: 'v2', name: 'Depth Anything V2 Small Q4', url: 'https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/64fe43eba7f8a384b02fe3fadaa26cbba35548d8/onnx/model_q4.onnx', size: 27404416},
  v2_base: {family: 'v2', name: 'Depth Anything V2 Base Q4', url: 'https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/dd4557d492cd7b563738ac8d9ccff9094620983c/onnx/model_q4.onnx', size: 102269714},
  v3: {family: 'v3', name: 'Depth Anything V3 Small INT8', url: 'https://huggingface.co/en970/depth-anything-v3-small-onnx/resolve/4cf0e2d05358b223ce873d1a696b3fb707c83c5d/onnx/model_quantized.onnx', size: 28941058},
  v3_base: {family: 'v3', fixedSide: 504, name: 'Depth Anything V3 Base FP16', url: 'https://huggingface.co/Heliosoph/da3-base-onnx/resolve/72080d285b91493e7e2f2f197d5e3afdf72ddb95/model_fp16.onnx', size: 207235996}
};

/** Sizes of the released ONNX files, used for offline space and transfer progress. */
const sizes = {
  ...Object.fromEntries(Object.values(depthProfiles).map(p => [p.url, p.size])),
  '../models/fcos_r50/cone_detector.onnx': 64569685,
  '../models/fcos_r18/cone_detector.onnx': 27592422,
  '../models/tracon_v12/cone_keypoints.onnx': 1471390,
  '../models/tracon_v12/yolox_v10.onnx': 3669893,
  '../models/tracon_v12/cone_detector.onnx': 8921756,
  '../models/tracon_v12/ssdlite_mixed.onnx': 8921756
};
const cacheName = 'cone-model-files-v1:' + new URL('.', import.meta.url).pathname;
export const mib = bytes => (bytes / 1048576).toFixed(1);

async function openStore() {
  try { return await globalThis.caches?.open(cacheName); }
  catch { return null; }
}

export async function modelStorage(profile) {
  const store = await openStore();
  const paths = [profile.model, profile.keypoints_model];
  const saved = new Set((await store?.keys() ?? []).map(request => request.url));
  const has = path => saved.has(new URL(path, import.meta.url).href);
  const sum = paths => paths.reduce((n, p) => n + sizes[p], 0);
  return {
    available: Boolean(store),
    saved: sum(Object.keys(sizes).filter(has)),
    missing: sum(paths.filter(p => !has(p)))
  };
}

export async function deleteModels() {
  await globalThis.caches?.delete(cacheName);
}

export async function readModels(models, confirmDownload, progress) {
  const store = await openStore();
  const files = await Promise.all(models.map(async ({path, label}) => {
    const url = new URL(path, import.meta.url).href;
    return {url, label, total: sizes[path], saved: await store?.match(url)};
  }));
  const missing = files.filter(file => !file.saved);
  if (missing.length && !await confirmDownload(missing)) return null;
  const buffers = [];
  for (const file of files) {
    buffers.push(await readFile(file, store, (...args) => progress(file.label, ...args)));
  }
  return buffers;
}

async function readFile({url, total, saved}, store, progress) {
  if (saved) {
    progress('saved', 0, total);
    return new Uint8Array(await saved.arrayBuffer());
  }
  progress('download', 0, total);
  const res = await fetch(url, {cache: 'no-store'});
  if (!res.ok) throw new Error('モデルを取得できません（HTTP ' + res.status + '）');
  const reader = res.body?.getReader();
  let bytes;
  if (reader) {
    bytes = new Uint8Array(total);
    let received = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      if (received + value.length > total) {
        await reader.cancel();
        throw new Error('モデルのサイズが配布情報と一致しません');
      }
      bytes.set(value, received);
      received += value.length;
      progress('download', received, total);
    }
    if (received !== total) throw new Error('モデルのダウンロードが途中で終了しました');
  } else {
    bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes.length !== total) throw new Error('モデルのサイズが配布情報と一致しません');
    progress('download', total, total);
  }
  try { await store?.put(url, new Response(bytes)); }
  catch { /* A full or unavailable cache still allows this scan. */ }
  return bytes;
}
