import {test} from 'node:test';
import assert from 'node:assert/strict';
import {buildCloud, measuredCones} from '../web_demo/depth_cloud.mjs';
import {encodeLas} from '../web_demo/las.mjs';

function fixture() {
  const width=13, height=5, data=new Uint8ClampedArray(width*height*4).fill(255);
  const raw=new Float32Array(width*height).fill(2);
  // Scaled, rectangular depth map. A sample at (2,2) is pixel (4.5,4.5) in the source.
  const objects=[2,6,10].map((x,i)=>{
    const z=[2,5,16][i];
    return {points3d_m:[[0,0,.7]],pose:{accepted:true,depth_m:z,t:[(2*x+.5-12.5)*z/10,.35,z],
      R:[[1,0,0],[0,0,-1],[0,1,0]]}};
  });
  objects.push({...objects[0],pose:{...objects[0].pose,accepted:false}});
  const result={image_size:[26,10],K:[[10,0,12.5],[0,10,4.5],[0,0,1]],objects};
  return {image:{width,height,data},raw,result};
}

test('both depth conventions use all accepted cone ratios and project pixel centres in camera-local metres',()=>{
  const {image,raw,result}=fixture();
  assert.equal(measuredCones(result.objects).length,3);
  raw[0]=4; image.data.set([17,55,93,255]);
  for (const profile of ['v2','v3']) {
    const cloud=buildCloud(image,raw,profile,result);
    const z=profile==='v3'?10:2.5;
    assert.equal(cloud.scale,profile==='v3'?2.5:10);
    assert.equal(cloud.anchors.length,3);
    assert.deepEqual(Array.from(cloud.xyz.slice(0,3)),[-1.2*z,z,.4*z]);
    assert.deepEqual(Array.from(cloud.colors.slice(0,3)),[17,55,93]);
  }
});

test('invalid samples, out-of-frame cones and unavailable calibration are handled',()=>{
  const {image,raw,result}=fixture();
  result.objects[2].pose.t[0]=1e4;
  raw[0]=Infinity; image.data[7]=0;
  for (const profile of ['v2','v3']) {
    const cloud=buildCloud(image,raw,profile,result);
    assert.equal(cloud.anchors.length,2);
    assert.equal(cloud.xyz.length/3,raw.length-2);
  }
  raw.fill(0);
  assert.throws(()=>buildCloud(image,raw,'v3',result),/2個/);
});

test('LAS 1.2 RGB records reproduce point coordinates, counts, and quantized bounds',()=>{
  const xyz=new Float32Array([-1.0007,2,.4,3.5632,5,-.9]);
  const colors=new Uint8Array([255,0,17,40,80,120]);
  const bytes=encodeLas({xyz,colors}), v=new DataView(bytes.buffer);
  assert.equal(new TextDecoder().decode(bytes.slice(0,4)),'LASF');
  assert.deepEqual([...bytes.slice(24,26)],[1,2]);
  assert.equal(v.getUint16(94,true),227); assert.equal(v.getUint32(96,true),227);
  assert.equal(bytes[104],2);assert.equal(v.getUint16(105,true),26);
  assert.equal(v.getUint32(107,true),2);assert.equal(v.getUint32(111,true),2);
  assert.equal(bytes.length,227+2*26);
  for (let axis=0;axis<3;axis++) {
    assert.equal(v.getFloat64(131+8*axis,true),.001);
    const coords=[0,1].map(i=>{
      const base=227+i*26;
      assert.equal(bytes[base+14],9);
      assert.equal(v.getUint16(base+20+2*axis,true),colors[3*i+axis]*256);
      const position=v.getInt32(base+4*axis,true)/1000;
      assert.ok(Math.abs(position-xyz[3*i+axis])<=.0005);
      return position;
    });
    assert.equal(v.getFloat64(179+16*axis,true),Math.max(...coords));
    assert.equal(v.getFloat64(187+16*axis,true),Math.min(...coords));
  }
});
