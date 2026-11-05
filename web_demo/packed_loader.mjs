/** Restore gzip/base64 assets from small static files. */
export async function loadPackedBinary(base, count = 5) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('このブラウザは圧縮モデルの展開に対応していません（DecompressionStreamが必要）');
  }
  const fragments = await Promise.all(Array.from({length:count},(_,i)=>{
    const path = `${base}/part-${String(i).padStart(3,'0')}.txt`;
    return fetch(path).then(response=>{
      if(!response.ok) throw new Error(`モデルのデータを取得できません (${response.status}): ${path}`);
      return response.text();
    });
  }));
  const encoded = fragments.join('').replace(/\s/g,'');
  const decoded = atob(encoded);
  const packed = new Uint8Array(decoded.length);
  for(let i=0;i<decoded.length;i++) packed[i] = decoded.charCodeAt(i);
  const stream = new Blob([packed],{type:'application/gzip'}).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function loadPackedScriptURL(base, count = 5) {
  const bytes = await loadPackedBinary(base,count);
  return URL.createObjectURL(new Blob([bytes],{type:'application/javascript'}));
}
