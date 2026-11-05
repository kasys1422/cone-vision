import test from 'node:test';
import assert from 'node:assert/strict';
import {readExifTags, estimateExifFov, readExifFov} from '../web_demo/exif_fov.mjs';

// Generate minimal EXIF/TIFF fixtures without depending on external test images.
function tiff({f35=26, focal=4.7, width=4032, height=3024, orientation=1, zoom=1,
                focalPlaneDensity=null, focalPlaneUnit=4, bigEndian=false}={}) {
  const le=!bigEndian;
  const entries0 = [[0x0112,3,orientation], [0x8769,4,0]];
  const exifEntries = [[0xa002,4,width], [0xa003,4,height],
    [0xa405,3,f35], [0x920a,5,focal], [0xa404,5,zoom]];
  if(focalPlaneDensity !== null) exifEntries.push([0xa20e,5,focalPlaneDensity], [0xa210,3,focalPlaneUnit]);
  const ifd0Offset=8, ifd1Offset=ifd0Offset+2+entries0.length*12+4;
  entries0[1][2]=ifd1Offset;
  let dataCursor=ifd1Offset+2+exifEntries.length*12+4;
  const arr=Buffer.alloc(500), b=new DataView(arr.buffer,arr.byteOffset,arr.length);
  arr.write(le?'II':'MM',0,'ascii');
  const u16=(pos,val)=>b.setUint16(pos,val,le);
  const u32=(pos,val)=>b.setUint32(pos,val,le);
  u16(2,42);u32(4,ifd0Offset);
  function writeIFD(offset, entries) {
    u16(offset,entries.length);
    entries.forEach(([tag,type,value],i)=>{
      const at=offset+2+i*12;
      u16(at,tag);u16(at+2,type);u32(at+4,1);
      if(type===3) u16(at+8,value);
      else if(type===4) u32(at+8,value);
      else if(type===5) {
        u32(at+8,dataCursor);
        u32(dataCursor,Math.round(value*100000));u32(dataCursor+4,100000);
        dataCursor+=8;
      }
    });
    u32(offset+2+entries.length*12,0);
  }
  writeIFD(ifd0Offset,entries0);
  writeIFD(ifd1Offset,exifEntries);
  return arr.subarray(0,dataCursor);
}
function jpeg(metadata=null) {
  const data=metadata?Buffer.concat([Buffer.from('Exif\0\0'),metadata]):Buffer.from('NonExif');
  const app1=Buffer.alloc(4);app1[0]=0xff;app1[1]=0xe1;app1.writeUInt16BE(data.length+2,2);
  return Buffer.concat([Buffer.from([0xff,0xd8]),app1,data,Buffer.from([0xff,0xd9])]);
}
function png(metadata) {
  const chunk=Buffer.alloc(8);chunk.writeUInt32BE(metadata.length,0);chunk.write('eXIf',4);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk,metadata,
    Buffer.alloc(4),Buffer.from([0,0,0,0,73,69,78,68,0,0,0,0])]);
}
function webp(metadata) {
  const c=Buffer.alloc(8);c.write('EXIF',0);c.writeUInt32LE(metadata.length,4);
  const payload=Buffer.concat([c,metadata,Buffer.alloc(metadata.length%2)]);
  const header=Buffer.alloc(12);header.write('RIFF',0);header.writeUInt32LE(payload.length+4,4);header.write('WEBP',8);
  return Buffer.concat([header,payload]);
}
const binary=buffer=>buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength);

for(const bigEndian of [false,true]) {
  test('JPEG EXIF '+(bigEndian?'big':'little')+' endian, 26mm equivalent',async()=>{
    const input=jpeg(tiff({bigEndian}));
    const tags=readExifTags(binary(input));
    assert.equal(tags.focalLength35mm,26);
    assert.equal(tags.imageWidth,4032);
    assert.equal(tags.imageHeight,3024);
    const result=estimateExifFov(tags,{width:4032,height:3024});
    assert.equal(result.status,'ok');
    assert.ok(result.fovDeg>67&&result.fovDeg<68);
    const read=await readExifFov(new Blob([input]),{width:4032,height:3024});
    assert.equal(read.status,'ok');
  });
}
test('portrait orientation uses displayed width',()=>{
  const tags=readExifTags(binary(jpeg(tiff({orientation:6}))));
  assert.equal(estimateExifFov(tags,{width:3024,height:4032}).status,'ok');
  assert.equal(estimateExifFov(tags,{width:4032,height:3024}).status,'unreliable');
});
test('digital zoom prevents automatic EXIF use',()=>{
  assert.equal(estimateExifFov(readExifTags(binary(jpeg(tiff({zoom:2})))),{width:4032,height:3024}).status,'unreliable');
});
test('aspect mismatch indicates crop and prevents auto use',()=>{
  assert.equal(estimateExifFov(readExifTags(binary(jpeg(tiff()))),{width:4032,height:2268}).status,'unreliable');
});
test('focal length alone is insufficient',()=>{
  assert.equal(estimateExifFov(readExifTags(binary(jpeg(tiff({f35:0})))),{width:4032,height:3024}).status,'missing');
});
test('focal-plane density fallback when available',()=>{
  const r=estimateExifFov(readExifTags(binary(jpeg(tiff({f35:0,focal:4,focalPlaneDensity:1000,focalPlaneUnit:4})))),{width:4032,height:3024});
  assert.equal(r.status,'ok');
  assert.ok(r.fovDeg>50&&r.fovDeg<55);
});
test('PNG and WebP EXIF container parsing',()=>{
  const metadata=tiff();
  for(const image of [png(metadata),webp(metadata)]){
    assert.equal(readExifTags(binary(image)).focalLength35mm,26);
    assert.equal(estimateExifFov(readExifTags(binary(image)),{width:4032,height:3024}).status,'ok');
  }
});
test('missing or damaged metadata fails safely',()=>{
  assert.equal(readExifTags(binary(jpeg())),null);
  assert.equal(readExifTags(binary(Buffer.from([0xff,0xd8,0xff,0xe1,0,18,1]))),null);
  assert.equal(estimateExifFov(null,{width:640,height:480}).status,'missing');
});
