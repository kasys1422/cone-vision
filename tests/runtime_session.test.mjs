import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createModelSession} from '../web_demo/runtime_session.mjs';

function runtime(t, fail = () => {}, gpu = {}) {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {configurable: true, value: {gpu}});
  t.after(() => {
    if (original) Object.defineProperty(globalThis, 'navigator', original);
    else delete globalThis.navigator;
  });
  const events = [];
  const ort = {InferenceSession: {async create(source, {executionProviders: [provider]}) {
    function record(stage) {
      events.push([source, provider, stage]);
      fail(source, provider, stage);
    }
    record('init');
    return {
      async run(...args) { record('run'); return args; },
      async release() { record('release'); }
    };
  }}};
  return {ort, events};
}

for (const stage of ['init', 'run']) {
  test('WebGPU ' + stage + ' failure retries only the affected model with the same feeds and fetches', async t => {
    const {ort, events} = runtime(t, (model, provider, action) => {
      if (model === 'depth' && provider === 'webgpu' && [stage, 'release'].includes(action)) throw Error('device lost');
    });
    const depth = await createModelSession(ort, 'depth');
    const detector = await createModelSession(ort, 'detector');
    const args = [{pixel_values: new Float32Array([3])}, ['predicted_depth']];
    const [first, second] = await Promise.all([depth.run(...args), depth.run(...args)]);
    assert.deepEqual(first, args);
    assert.deepEqual(second, args);
    await detector.run();
    assert.equal(depth.provider, 'wasm');
    assert.equal(detector.provider, 'webgpu');
    assert.deepEqual(events.filter(([model,provider]) => model === 'depth' && provider === 'wasm')
      .map(([, , action]) => action), ['init', 'run', 'run']);
    await Promise.all([depth.release(), detector.release()]);
  });
}

for (const [setting, gpu] of [[false, {}], [true, null]]) {
  test(setting ? 'missing WebGPU starts directly with WASM' : 'opt-out starts directly with WASM', async t => {
    const {ort, events} = runtime(t, (_model,_provider,stage) => {
      if (stage === 'run') throw Error('WASM failed');
    }, gpu);
    const session = await createModelSession(ort, 'depth', setting);
    await assert.rejects(session.run({}), /WASM failed/);
    await session.release();
    assert.deepEqual(events.map(([,provider,stage]) => [provider,stage]),
      [['wasm','init'], ['wasm','run'], ['wasm','release']]);
  });
}
