/** Sizes of the released ONNX files, used for offline space and transfer progress. */
const sizes = {
  '../models/fcos_r50/cone_detector.onnx': 64569685,
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

export async function readModel(path, progress) {
  const url = new URL(path, import.meta.url).href;
  const store = await openStore();
  const saved = await store?.match(url);
  if (saved) {
    progress('saved', 0, sizes[path]);
    return new Uint8Array(await saved.arrayBuffer());
  }
  const total = sizes[path];
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
