/**
 * EXIF focal-length reader for Cone Vision (GPL-3.0-only).
 * Only the camera tags necessary to estimate field of view are decoded;
 * GPS, serial numbers, and other private EXIF metadata are not collected.
 * Supported containers: JPEG APP1 Exif, PNG eXIf, WebP EXIF.
 */
const DIAGONAL_35MM = Math.hypot(36, 24);
const MAX_EXIF_SCAN = 2 * 1024 * 1024;

const TYPE_LENGTH = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8};
const TIFF_TAGS = {
  imageWidth: 0x0100,
  imageHeight: 0x0101,
  orientation: 0x0112,
  exifPointer: 0x8769,
  focalLengthMm: 0x920a,
  digitalZoomRatio: 0xa404,
  focalLength35mm: 0xa405,
  pixelXDimension: 0xa002,
  pixelYDimension: 0xa003,
  focalPlaneXResolution: 0xa20e,
  focalPlaneYResolution: 0xa20f,
  focalPlaneResolutionUnit: 0xa210
};

function ascii(view, start, length) {
  if (start < 0 || start + length > view.byteLength) return '';
  let out = '';
  for (let i = 0; i < length; i++) out += String.fromCharCode(view.getUint8(start + i));
  return out;
}

function parseTiff(view, start, limit = view.byteLength) {
  const end = Math.min(limit, view.byteLength);
  if (start < 0 || start + 8 > end) return null;
  const mark = ascii(view, start, 2);
  if (mark !== 'II' && mark !== 'MM') return null;
  const le = mark === 'II';
  const u16 = off => off + 2 <= end ? view.getUint16(off, le) : null;
  const u32 = off => off + 4 <= end ? view.getUint32(off, le) : null;
  const s32 = off => off + 4 <= end ? view.getInt32(off, le) : null;
  if (u16(start + 2) !== 42) return null;
  const firstIFD = u32(start + 4);
  if (firstIFD === null) return null;

  function parseIFD(relativeOffset) {
    if (!Number.isInteger(relativeOffset)) return {};
    const off = start + relativeOffset;
    if (off < start || off + 2 > end) return {};
    const count = u16(off);
    if (count === null || count > 512 || off + 2 + 12 * count > end) return {};
    const entries = {};
    for (let i = 0; i < count; i++) {
      const entry = off + 2 + i * 12;
      const tag = u16(entry), type = u16(entry + 2), n = u32(entry + 4);
      if (tag === null || !TYPE_LENGTH[type] || n !== 1) continue;
      const size = TYPE_LENGTH[type] * n;
      const location = size <= 4 ? entry + 8 : start + u32(entry + 8);
      if (location < start || location + size > end) continue;
      let value = null;
      if (type === 3) value = u16(location);
      else if (type === 4) value = u32(location);
      else if (type === 9) value = s32(location);
      else if (type === 5 || type === 10) {
        const numerator = type === 5 ? u32(location) : s32(location);
        const denominator = type === 5 ? u32(location + 4) : s32(location + 4);
        if (numerator !== null && denominator) value = numerator / denominator;
      }
      if (value !== null && Number.isFinite(value)) entries[tag] = value;
    }
    return entries;
  }

  const ifd0 = parseIFD(firstIFD);
  const exif = ifd0[TIFF_TAGS.exifPointer] ? parseIFD(ifd0[TIFF_TAGS.exifPointer]) : {};
  const val = (name) => exif[TIFF_TAGS[name]] ?? ifd0[TIFF_TAGS[name]] ?? null;
  return {
    focalLength35mm: val('focalLength35mm'),
    focalLengthMm: val('focalLengthMm'),
    digitalZoomRatio: val('digitalZoomRatio'),
    orientation: ifd0[TIFF_TAGS.orientation] ?? 1,
    imageWidth: val('pixelXDimension') ?? val('imageWidth'),
    imageHeight: val('pixelYDimension') ?? val('imageHeight'),
    focalPlaneXResolution: val('focalPlaneXResolution'),
    focalPlaneYResolution: val('focalPlaneYResolution'),
    focalPlaneResolutionUnit: val('focalPlaneResolutionUnit')
  };
}

/** Parse only the EXIF TIFF portion, ignoring unrelated metadata. */
export function readExifTags(buffer) {
  const view = new DataView(buffer);
  const n = view.byteLength;
  if (n < 12) return null;
  // JPEG SOI (APP1 segments may come after APP0, XMP, thumbnail, etc.)
  if (view.getUint16(0) === 0xffd8) {
    let cursor = 2;
    while (cursor + 4 < n) {
      if (view.getUint8(cursor) !== 0xff) break;
      while (cursor < n && view.getUint8(cursor) === 0xff) cursor++;
      if (cursor >= n) break;
      const marker = view.getUint8(cursor++);
      if (marker === 0xda || marker === 0xd9) break; // image scan / EOI
      if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
      if (cursor + 2 > n) break;
      const length = view.getUint16(cursor, false);
      if (length < 2 || cursor + length > n) break;
      const payload = cursor + 2;
      if (marker === 0xe1 && ascii(view, payload, 6) === 'Exif\x00\x00') {
        const result = parseTiff(view, payload + 6, cursor + length);
        if (result) return result;
      }
      cursor += length;
    }
    return null;
  }
  // PNG eXIf chunk (EXIF TIFF data without the Exif\0\0 prefix).
  if (ascii(view, 1, 3) === 'PNG' && view.getUint8(0) === 0x89) {
    let cursor = 8;
    while (cursor + 12 <= n) {
      const length = view.getUint32(cursor, false);
      const type = ascii(view, cursor + 4, 4);
      const end = cursor + 8 + length;
      if (end + 4 > n) break;
      if (type === 'eXIf') return parseTiff(view, cursor + 8, end);
      if (type === 'IEND') break;
      cursor = end + 4;
    }
    return null;
  }
  // WebP EXIF chunk.
  if (ascii(view, 0, 4) === 'RIFF' && ascii(view, 8, 4) === 'WEBP') {
    let cursor = 12;
    while (cursor + 8 <= n) {
      const type = ascii(view, cursor, 4);
      const length = view.getUint32(cursor + 4, true);
      const payload = cursor + 8;
      if (payload + length > n) break;
      if (type === 'EXIF') {
        const prefix = ascii(view, payload, 6) === 'Exif\x00\x00' ? 6 : 0;
        return parseTiff(view, payload + prefix, payload + length);
      }
      cursor = payload + length + (length & 1);
    }
  }
  return null;
}

function dimensionsMismatch(meta, width, height) {
  if (!meta.imageWidth || !meta.imageHeight) return false;
  const rotated = [5, 6, 7, 8].includes(meta.orientation);
  const expected = rotated ? meta.imageHeight / meta.imageWidth : meta.imageWidth / meta.imageHeight;
  const actual = width / height;
  // Aspect mismatch indicates crop, rotation/metadata mismatch, or EXIF inconsistency.
  return Math.abs(Math.log(actual / expected)) > 0.025;
}

const UNIT_MM = {2: 25.4, 3: 10, 4: 1, 5: 0.001};
/**
 * Compute approximate pinhole intrinsics for decoded bitmap dimensions.
 * FocalLengthIn35mmFilm is defined for a 36x24mm full-frame diagonal.
 * No principal point or distortion calibration is implied by this estimate.
 */
export function estimateExifFov(meta, {width, height}) {
  const empty = (status, message) => ({status, message, cameraMode: '手動概算'});
  if (!(width > 0 && height > 0)) return empty('unreliable', '読み込み画像のサイズが不正です');
  if (!meta) return empty('missing', 'EXIFの焦点距離が見つかりません。画角は手動で設定してください。');
  if (dimensionsMismatch(meta, width, height)) {
    return empty('unreliable', 'EXIFと画像の縦横比が異なります。トリミングの可能性があるため自動適用しません。');
  }
  if (meta.digitalZoomRatio && meta.digitalZoomRatio > 1.02) {
    return empty('unreliable', 'デジタルズーム ' + meta.digitalZoomRatio.toFixed(2) + '倍のため自動適用しません。');
  }
  let focalPx = null, source = null;
  const eq35 = meta.focalLength35mm;
  if (Number.isFinite(eq35) && eq35 > 0 && eq35 < 2000) {
    focalPx = (eq35 / DIAGONAL_35MM) * Math.hypot(width, height);
    source = '35mm判換算';
  } else {
    // Optional fallback if BOTH physical focal length AND focal-plane density exist.
    // Do not guess sensor size from camera model or FocalLength alone.
    const unitMm = UNIT_MM[meta.focalPlaneResolutionUnit];
    const f = meta.focalLengthMm;
    const density = meta.focalPlaneXResolution;
    if (unitMm && f > 0 && f < 1000 && density > 0 && density < 1e8 && meta.imageWidth > 0 && meta.imageHeight > 0) {
      // Density is in pixels per physical unit of the *original* uncropped raster.
      // Use its geometric scaling to the decoded image diagonal.
      const rawWidth = meta.imageWidth, rawHeight = meta.imageHeight;
      const originalFocalPixels = f * density / unitMm;
      focalPx = originalFocalPixels * Math.hypot(width, height) / Math.hypot(rawWidth, rawHeight);
      source = 'センサー画素密度';
    }
  }
  if (!(focalPx > 0 && Number.isFinite(focalPx))) {
    return empty('missing', '画角の推定に必要なEXIF情報がありません。35mm判換算焦点距離が必要です。');
  }
  const fovDeg = 2 * Math.atan(width / (2 * focalPx)) * 180 / Math.PI;
  if (!(fovDeg > 4 && fovDeg < 175)) return empty('unreliable', 'EXIFから求めた画角が不自然な値でした。');
  return {
    status: 'ok',
    source,
    cameraMode: 'EXIF概算',
    fovDeg,
    focalPx,
    focalLength35mm: eq35 ?? null,
    focalLengthMm: meta.focalLengthMm ?? null,
    orientation: meta.orientation,
    width,
    height,
    message: source + ' ' + (source === '35mm判換算' ? (eq35 + ' mm') : '') + ' → 水平画角 約' + fovDeg.toFixed(1) + '°（概算）'
  };
}

export async function readExifFov(file, bitmapSize) {
  if (!file || typeof file.slice !== 'function') return estimateExifFov(null, bitmapSize);
  try {
    const data = await file.slice(0, Math.min(file.size, MAX_EXIF_SCAN)).arrayBuffer();
    const tags = readExifTags(data);
    return estimateExifFov(tags, bitmapSize);
  } catch {
    return estimateExifFov(null, bitmapSize);
  }
}
