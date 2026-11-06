import {createModelSession} from './runtime_session.mjs';
import {depthProfiles} from './model_store.mjs';

/** PnP uses OpenCV camera axes. The point cloud uses right, forward, up (metres). */
const local = ([x, y, z]) => [x, z, -y];
export function median(values) {
  const a = [...values].sort((x, y) => x - y);
  const i = Math.floor(a.length / 2);
  return a.length % 2 ? a[i] : (a[i - 1] + a[i]) / 2;
}

export function measuredCones(objects) {
  return objects.filter(o => o.pose?.accepted && o.pose.depth_m > 0 && Number.isFinite(o.pose.depth_m));
}

function depthImage(bitmap, result, cv, side) {
  const canvas = document.createElement('canvas');
  const longEdge = Math.max(bitmap.width, bitmap.height);
  const size = span => 14 * Math.max(1, Math.round(side / 14 * span / longEdge));
  canvas.width = size(bitmap.width);
  canvas.height = size(bitmap.height);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  if (!result.distortion?.some(x => x !== 0)) return image;

  // Rectify RGB and depth together. Alpha marks rays outside the source photo.
  const sx = canvas.width / bitmap.width, sy = canvas.height / bitmap.height;
  const K = result.K;
  const mats = [];
  const own = m => { mats.push(m); return m; };
  try {
    const camera = own(cv.matFromArray(3, 3, cv.CV_64F, [
      K[0][0]*sx, 0, (K[0][2]+0.5)*sx-0.5,
      0, K[1][1]*sy, (K[1][2]+0.5)*sy-0.5, 0, 0, 1
    ]));
    const dist = own(cv.matFromArray(result.distortion.length, 1, cv.CV_64F, result.distortion));
    const src = own(cv.matFromImageData(image)), dst = own(new cv.Mat());
    cv.undistort(src, dst, camera, dist, camera);
    image.data.set(dst.data);
    return image;
  } finally { mats.forEach(m => m.delete()); }
}

export async function inferCloud(ort, bytes, profile, bitmap, result, cv, preferWebGPU = true) {
  const spec = depthProfiles[profile];
  const image = depthImage(bitmap, result, cv, spec.fixedSide ?? 518);
  const raw = await inferDepthImage(ort, bytes, spec, image, preferWebGPU);
  return buildCloud(image, raw, spec.family, result);
}

export async function inferDepthImage(ort, bytes, {family, fixedSide}, image, preferWebGPU) {
  const {width, height, data} = image;
  // Single-view fixed-square exports need padding. Keep it outside the depth and RGB maps.
  const inputW = fixedSide ?? width, inputH = fixedSide ?? height;
  const left = Math.floor((inputW-width)/2), top = Math.floor((inputH-height)/2);
  const n = inputW * inputH;
  const rgb = new Float32Array(3*n);
  const mean = [.485, .456, .406], std = [.229, .224, .225];
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      const row = c*n + (y+top)*inputW + left;
      for (let x = 0; x < width; x++) rgb[row+x] = (data[4*(y*width+x)+c]/255 - mean[c])/std[c];
    }
  }
  const session = await createModelSession(ort, bytes, preferWebGPU);
  const dims = family === 'v3' && !fixedSide ? [1,1,3,inputH,inputW] : [1,3,inputH,inputW];
  const inputName = fixedSide ? 'image' : 'pixel_values';
  const outputName = fixedSide ? 'depth' : 'predicted_depth';
  let tensor, output;
  try {
    tensor = new ort.Tensor('float32', rgb, dims);
    output = (await session.run({[inputName]: tensor}, [outputName]))[outputName];
    if (output.dims.at(-1) !== inputW || output.dims.at(-2) !== inputH) throw Error('深度画像の寸法が一致しません');
    let depth = output.data;
    if (width !== inputW || height !== inputH) {
      depth = new Float32Array(width*height);
      for (let y = 0; y < height; y++) {
        const start = (top+y)*inputW + left;
        depth.set(output.data.subarray(start,start+width), y*width);
      }
    }
    return depth;
  } finally {
    output?.dispose();
    tensor?.dispose();
    await session.release();
  }
}

export function buildCloud(image, raw, family, result) {
  const {width, height, data} = image;
  const [w,h] = result.image_size;
  const sx = width/w, sy = height/h;
  const [[fx,,cx],[,fy,cy]] = result.K;
  const anchors = [];
  for (const obj of measuredCones(result.objects)) {
    const {R,t} = obj.pose;
    const up = R.map(row => row[2]);
    const half = obj.points3d_m[0][2]/2;
    const [x,y,z] = t.map((v,i) => v + up[i]*half);
    if (!(z > 0)) continue;
    const u = Math.round((fx*x/z+cx+.5)*sx-.5);
    const v = Math.round((fy*y/z+cy+.5)*sy-.5);
    if (u<0 || v<0 || u>=width || v>=height) continue;
    const patch = [];
    for (let j=Math.max(0,v-1); j<=Math.min(height-1,v+1); j++) {
      for (let k=Math.max(0,u-1); k<=Math.min(width-1,u+1); k++) {
        const i=j*width+k;
        if (data[i*4+3] === 255 && raw[i]>0 && Number.isFinite(raw[i])) patch.push(raw[i]);
      }
    }
    if (!patch.length) continue;
    const d = median(patch);
    anchors.push({base: local(t), up: local(up), z, ratio: family === 'v2' ? z*d : z/d});
  }
  if (anchors.length < 2) throw Error('深度と対応するコーンが2個必要です');
  const scale = median(anchors.map(a => a.ratio));
  // Exclude unsupported far background; keep every anchor inside the depth range.
  const far = Math.max(40, 3*Math.max(...anchors.map(a => a.z)));
  const xyz = new Float32Array(raw.length*3), colors = new Uint8Array(raw.length*3);
  const metres = new Float32Array(raw.length);
  let count = 0;
  for (let i=0; i<raw.length; i++) {
    if (data[i*4+3] !== 255 || !(raw[i]>0)) continue;
    const z = family === 'v2' ? scale/raw[i] : scale*raw[i];
    if (!(z>0 && z<=far)) continue;
    metres[i] = z;
    const u = (i%width+.5)/sx-.5, v = (Math.floor(i/width)+.5)/sy-.5;
    const j = 3*count++;
    xyz[j] = (u-cx)*z/fx;
    xyz[j+1] = z;
    xyz[j+2] = -(v-cy)*z/fy;
    colors[j] = data[4*i]; colors[j+1] = data[4*i+1]; colors[j+2] = data[4*i+2];
  }
  if (!count) throw Error('点群に変換できる画素がありません');
  return {xyz: xyz.subarray(0,3*count), colors: colors.subarray(0,3*count), anchors, scale, far,
    depthMap: {metres, width, height}, colorImage: image};
}
