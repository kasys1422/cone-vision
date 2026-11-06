/** Keep the retry local to the failed model. Inputs are CPU tensors and can be replayed. */
export async function createModelSession(ort, source, preferWebGPU = true) {
  let provider = preferWebGPU && globalThis.navigator?.gpu ? 'webgpu' : 'wasm';
  async function create() {
    const next = await ort.InferenceSession.create(source, {executionProviders: [provider]});
    // Only a live GPU session still needs the model bytes for a later retry.
    if (provider === 'wasm') source = null;
    return next;
  }
  let session;
  async function switchToWasm(error) {
    if (provider !== 'webgpu') throw error;
    console.warn('WebGPU session failed; retrying with WASM.', error);
    provider = 'wasm';
    const previous = session;
    session = null;
    // A lost GPU device can also reject release().
    try { await previous?.release(); } catch {}
    session = await create();
  }
  try { session = await create(); }
  catch (error) { await switchToWasm(error); }

  // Serialize use and release so that overlapping requests cannot replace an in-use session.
  let pending = Promise.resolve();
  let closed = false;
  function enqueue(task) {
    const result = pending.then(task);
    pending = result.catch(() => {});
    return result;
  }
  return {
    get provider() { return provider; },
    run(...args) {
      return enqueue(async () => {
        if (closed) throw Error('推論セッションは終了しました');
        // A previous WASM initialization may have failed transiently.
        session ??= await create();
        try { return await session.run(...args); }
        catch (error) {
          await switchToWasm(error);
          return session.run(...args);
        }
      });
    },
    release() {
      return enqueue(async () => {
        closed = true;
        source = null;
        const previous = session;
        session = null;
        await previous?.release();
      });
    }
  };
}
