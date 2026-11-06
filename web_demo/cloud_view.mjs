import {median} from './depth_cloud.mjs';

const dot = (a,b) => a.reduce((sum,v,i) => sum+v*b[i],0);
const cross = ([a,b,c],[x,y,z]) => [b*z-c*y,c*x-a*z,a*y-b*x];
const unit = a => a.map(v => v/(Math.hypot(...a)||1));
const clamp = (v,lo,hi) => Math.max(lo,Math.min(v,hi));

function makeGrid(anchors, center, extent) {
  const normal = unit([0,1,2].map(i => median(anchors.map(a => a.up[i]))));
  const right = unit(cross(Math.abs(normal[1])<.9 ? [0,1,0] : [0,0,1], normal));
  const forward = unit(cross(normal,right));
  const offset = median(anchors.map(a => dot(a.base,normal))) - dot(center,normal);
  const origin = center.map((v,i) => v + offset*normal[i]);
  const xyz=[], colors=[];
  function vertex(x,y,major) {
    xyz.push(...origin.map((v,i) => v+x*right[i]+y*forward[i]));
    colors.push(...(major ? [170,206,216] : [97,131,143]));
  }
  for (let k=-extent; k<=extent; k++) {
    const major = k%5===0;
    vertex(k,-extent,major); vertex(k,extent,major);
    vertex(-extent,k,major); vertex(extent,k,major);
  }
  return {xyz:new Float32Array(xyz), colors:new Uint8Array(colors), normal};
}

// Camera basis and perspective are evaluated on the GPU; redraw only after a gesture or resize.
export function openViewer(canvas, cloud, gridInput) {
  const gl = canvas.getContext('webgl', {alpha:false, antialias:true});
  if (!gl || gl.isContextLost()) throw Error('この端末で3D表示を開始できません');
  const center = [0,1,2].map(i => median(cloud.anchors.map(a => a.base[i])));
  const distance = Math.max(2,center[1]*1.25, ...cloud.anchors.map(a=>Math.abs(a.base[0]-center[0])*2));
  const grid = makeGrid(cloud.anchors,center,Math.min(60,Math.max(8,Math.ceil(distance*1.5))));
  const shaders=[], buffers=[], program=gl.createProgram();
  const events=new AbortController(), options={signal:events.signal};
  let frame=0, observer;
  function close() {
    events.abort(); observer?.disconnect(); cancelAnimationFrame(frame);
    gl.useProgram(null); buffers.forEach(b=>gl.deleteBuffer(b));
    gl.deleteProgram(program); shaders.forEach(s=>gl.deleteShader(s));
  }
  function shader(type,src) {
    const s=gl.createShader(type); shaders.push(s);
    gl.shaderSource(s,src); gl.compileShader(s);
    if (!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw Error('3D描画の準備に失敗しました');
    gl.attachShader(program,s);
  }
  function buffer(array) {
    const b=gl.createBuffer(); buffers.push(b);
    gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,array,gl.STATIC_DRAW);
    return b;
  }
  try {
    shader(gl.VERTEX_SHADER,`attribute vec3 pos, rgb;
      uniform vec3 target, right, up, front;
      uniform float radius, aspect, size, clipFar;
      varying vec3 color;
      void main() {
        vec3 p=pos-target;
        float z=dot(p,front)+radius;
        float near=0.03;
        gl_Position=vec4(dot(p,right)*1.8/aspect, dot(p,up)*1.8,
          ((clipFar+near)*z-2.0*clipFar*near)/(clipFar-near), z);
        gl_PointSize=size; color=rgb;
      }`);
    shader(gl.FRAGMENT_SHADER,`precision mediump float; varying vec3 color;
      void main(){gl_FragColor=vec4(color,1.0);}`);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error('3D表示を初期化できません');
    const upload = obj => ({pos:buffer(obj.xyz),rgb:buffer(obj.colors),count:obj.xyz.length/3});
    const points=upload(cloud), lines=upload(grid);
    const pos=gl.getAttribLocation(program,'pos'), rgb=gl.getAttribLocation(program,'rgb');
    const loc=Object.fromEntries(['target','right','up','front','radius','aspect','size','clipFar']
      .map(key=>[key,gl.getUniformLocation(program,key)]));
    let yaw, pitch, radius, target;
    function draw() {
      frame=0;
      const dpr=Math.min(window.devicePixelRatio||1,2);
      const w=Math.max(1,Math.round(canvas.clientWidth*dpr)), h=Math.max(1,Math.round(canvas.clientHeight*dpr));
      if (canvas.width!==w || canvas.height!==h) {canvas.width=w; canvas.height=h;}
      gl.viewport(0,0,w,h); gl.clearColor(.082,.108,.125,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
      const front=[Math.sin(yaw)*Math.cos(pitch), Math.cos(yaw)*Math.cos(pitch), -Math.sin(pitch)];
      const right=[Math.cos(yaw),-Math.sin(yaw),0];
      gl.uniform3fv(loc.target,target); gl.uniform3fv(loc.front,front);
      gl.uniform3fv(loc.right,right); gl.uniform3fv(loc.up,cross(right,front));
      gl.uniform1f(loc.radius,radius); gl.uniform1f(loc.aspect,w/h);
      gl.uniform1f(loc.clipFar,radius+cloud.far*8+100);
      gl.uniform1f(loc.size,clamp(dpr*Math.max(1.35,canvas.clientHeight/350),1,6));
      function plot(obj,mode) {
        gl.bindBuffer(gl.ARRAY_BUFFER,obj.pos); gl.enableVertexAttribArray(pos);
        gl.vertexAttribPointer(pos,3,gl.FLOAT,false,0,0);
        gl.bindBuffer(gl.ARRAY_BUFFER,obj.rgb); gl.enableVertexAttribArray(rgb);
        gl.vertexAttribPointer(rgb,3,gl.UNSIGNED_BYTE,true,0,0);
        gl.drawArrays(mode,0,obj.count);
      }
      plot(points,gl.POINTS);
      if (gridInput.checked) plot(lines,gl.LINES);
    }
    function redraw() { if (!frame) frame=requestAnimationFrame(draw); }
    function reset() { yaw=.13; pitch=.16; radius=distance; target=[...center]; redraw(); }
    function zoom(factor, startRadius=radius) { radius=clamp(startRadius*factor,.2,Math.max(100,distance*12)); redraw(); }
    function pan(dx,dy, startTarget=target, startRadius=radius) {
      const right=[Math.cos(yaw),-Math.sin(yaw),0];
      const heading=[Math.sin(yaw),Math.cos(yaw),0];
      const normal=grid.normal;
      const perpendicular=dot(right,normal);
      const projected=right.map((v,i)=>v-perpendicular*normal[i]);
      const along=Math.hypot(...projected)>1e-5 ? unit(projected) : unit(cross(heading,normal));
      const across=cross(normal,along);
      const sign=dot(across,heading)<0 ? -1 : 1;
      const metresPerPixel=2*startRadius/(1.8*Math.max(1,canvas.clientHeight));
      target=startTarget.map((v,i)=>v+metresPerPixel*(-dx*along[i]+dy*sign*across[i]));
      redraw();
    }
    const fingers=new Map();
    let gesture=null;
    function touchPair() {
      const [a,b]=fingers.values();
      return {x:(a[0]+b[0])/2, y:(a[1]+b[1])/2, span:Math.max(1,Math.hypot(a[0]-b[0],a[1]-b[1]))};
    }
    function rebaseGesture() {
      gesture=fingers.size===2 ? {...touchPair(), target, radius} : null;
    }
    canvas.addEventListener('pointerdown',e=>{
      if (e.pointerType==='mouse' && e.button===1) e.preventDefault();
      else if (e.button!==0) return;
      canvas.setPointerCapture(e.pointerId); fingers.set(e.pointerId,[e.clientX,e.clientY]);
      rebaseGesture();
    },options);
    canvas.addEventListener('pointermove',e=>{
      const old=fingers.get(e.pointerId);
      if (!old) return;
      fingers.set(e.pointerId,[e.clientX,e.clientY]);
      if (e.pointerType==='mouse' && (e.buttons & 4)) {
        pan(e.clientX-old[0],e.clientY-old[1]);
        return;
      }
      if (e.pointerType==='mouse' && !(e.buttons & 1)) return;
      if (gesture) {
        const pair=touchPair();
        // Both transforms use the same starting pose, regardless of pointer event order.
        pan(pair.x-gesture.x,pair.y-gesture.y,gesture.target,gesture.radius);
        zoom(gesture.span/pair.span,gesture.radius);
      }
      if (fingers.size===1) {
        yaw+=(e.clientX-old[0])*.008; pitch=clamp(pitch+(e.clientY-old[1])*.008,-1.48,1.48); redraw();
      }
    },options);
    for (const name of ['pointerup','pointercancel','lostpointercapture']) {
      canvas.addEventListener(name,e=>{
        if (fingers.delete(e.pointerId)) rebaseGesture();
      },options);
    }
    canvas.addEventListener('auxclick',e=>{ if (e.button===1) e.preventDefault(); },options);
    canvas.addEventListener('wheel',e=>{
      e.preventDefault(); zoom(Math.exp(clamp(e.deltaY*.002,-1,1)));
    },{...options,passive:false});
    canvas.addEventListener('keydown',e=>{
      switch(e.key) {
        case 'ArrowLeft': yaw-=.12; break;
        case 'ArrowRight': yaw+=.12; break;
        case 'ArrowUp': pitch=clamp(pitch+.12,-1.48,1.48); break;
        case 'ArrowDown': pitch=clamp(pitch-.12,-1.48,1.48); break;
        case '+': case '=': zoom(.8); break;
        case '-': zoom(1.25); break;
        default: return;
      }
      e.preventDefault(); redraw();
    },options);
    gridInput.addEventListener('change',redraw,options);
    observer=new ResizeObserver(redraw); observer.observe(canvas);
    reset();
    return {close,reset};
  } catch (error) { close(); throw error; }
}
