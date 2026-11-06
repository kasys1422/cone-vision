import test from 'node:test';
import assert from 'node:assert/strict';
import {ConeVision, decodeYolox, sameCone, ssdInput} from '../web_demo/inference.mjs';
import {poseSettingsForRoi} from '../web_demo/pnp.mjs';

test('SSD stretches non-square input, interpolates in float and preserves RGB order', () => {
  const image = {width: 2, height: 1, data: Uint8Array.from([0, 10, 80, 255, 5, 30, 20, 255])};
  const {tensor, transform} = ssdInput(image, 4);
  assert.equal(transform.sx, 2);
  assert.equal(transform.sy, 4);
  const expected = [[0, 1.25, 3.75, 5], [10, 15, 25, 30], [80, 65, 35, 20]];
  for (const [channel, row] of expected.entries()) {
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        assert.ok(Math.abs(tensor[channel * 16 + y * 4 + x] * 255 - row[x]) < 0.00001);
      }
    }
  }
});

test('SSD boxes map back to the photographed image after suppression', () => {
  const transform = {sx: 0.5, sy: 1, left: 0, top: 0, originalWidth: 640, originalHeight: 320};
  const predictions = [40, 60, 20, 40, .9, 1, 40, 60, 20, 40, .7, 1, 120, 120, 8, 16, .4, 1];
  const boxes = decodeYolox(predictions, transform, .5, .5, 300, 0);
  assert.deepEqual(boxes.map(b => b.bbox_xyxy), [[60, 40, 100, 80]]);
});

test('resampling the same ROI keeps the previously calibrated PnP tolerance', () => {
  const previous = poseSettingsForRoi({side: 384, size: 96});
  assert.deepEqual(poseSettingsForRoi({side: 384, size: 192}), previous);
});

test('a detector crop is translated back before sampling keypoints and composing the pose inputs', async () => {
  const image = {width: 40, height: 32, data: new Uint8ClampedArray(40*32*4)};
  const region = [20, 10, 28, 18];
  for (let y = 10; y < 18; y++) for (let x = 20; x < 28; x++) image.data[4*(y*40+x)] = 255;
  class Tensor { constructor(type, data) { this.data = data; } }
  const detector = {run({images}) {
    assert.equal(images.data[0], 1);
    return {predictions: {data: [4,4,4,4,.9,1]}};
  }};
  const keypoints = {run({images}) {
    assert.equal(images.data[0], 1);
    return {coordinates: {data: Array(10).fill(.5)}, confidence: {data: Array(5).fill(1)},
      visibility: {data: Array(15).fill(1)}};
  }};
  const vision = new ConeVision({Tensor}, detector, keypoints, 'wasm', {size: 8, keypoint_size: 4});
  const K = [[30,0,20],[0,30,16],[0,0,1]];
  const result = await vision.predict(image, {region, K});
  assert.deepEqual(result.image_size, [40,32]);
  const [obj] = result.objects;
  assert.deepEqual(obj.bbox_xyxy, [22,12,26,16]);
  assert.deepEqual(obj.pnp_input.points2d, Array(5).fill([24,14]));
  assert.deepEqual(obj.pnp_input.K, K);
});

test('crop duplicates can have tighter boxes, while neighboring cones stay separate', () => {
  const box = [100,100,120,140];
  assert.ok(sameCone(box, box));
  assert.ok(sameCone(box, [104,110,116,130]));
  assert.ok(!sameCone(box, [118,102,138,142]));
});
