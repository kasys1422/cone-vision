import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeYolox, ssdInput} from '../web_demo/inference.mjs';
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
