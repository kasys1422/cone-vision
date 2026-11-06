import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {extractReleaseModel} from '../web_demo/model_store.mjs';

test('release model extraction checks the stored member, size and SHA-256', async () => {
  const payload = Uint8Array.of(4, 5, 6);
  const name = new TextEncoder().encode('model.onnx');
  const bytes = new Uint8Array(30 + name.length + payload.length);
  const header = new DataView(bytes.buffer);
  header.setUint32(0, 0x04034b50, true);
  header.setUint32(18, payload.length, true);
  header.setUint32(22, payload.length, true);
  header.setUint16(26, name.length, true);
  bytes.set(name, 30);
  bytes.set(payload, 30 + name.length);
  const spec = {size: payload.length, zipSize: bytes.length, sha256: createHash('sha256').update(payload).digest('hex')};
  assert.deepEqual(await extractReleaseModel(bytes, spec), payload);
  await assert.rejects(extractReleaseModel(bytes.subarray(0, -1), spec), /サイズ/);
  bytes[bytes.length - 1] ^= 1;
  await assert.rejects(extractReleaseModel(bytes, spec), /SHA-256/);
  header.setUint16(8, 8, true);
  await assert.rejects(extractReleaseModel(bytes, spec), /形式/);
});
