/** LAS 1.2 point format 2, local metre coordinates (right, forward, up). */
export function encodeLas({xyz, colors}) {
  const count = xyz.length/3;
  if (!Number.isInteger(count) || !count || colors.length !== xyz.length) throw Error('点群が空か、色の数が一致しません');
  const header = 227, record = 26;
  const bytes = new Uint8Array(header+count*record);
  const view = new DataView(bytes.buffer);
  const text = (offset,value) => bytes.set(new TextEncoder().encode(value),offset);
  text(0,'LASF'); bytes[24]=1; bytes[25]=2;
  text(26,'Cone Vision camera-local metres'); text(58,'Cone Vision');
  const date=new Date();
  view.setUint16(90,1+Math.floor((Date.UTC(date.getUTCFullYear(),date.getUTCMonth(),date.getUTCDate())-Date.UTC(date.getUTCFullYear(),0,1))/86400000),true);
  view.setUint16(92,date.getUTCFullYear(),true);
  view.setUint16(94,header,true); view.setUint32(96,header,true);
  bytes[104]=2; view.setUint16(105,record,true);
  view.setUint32(107,count,true); view.setUint32(111,count,true);
  const min=[Infinity,Infinity,Infinity], max=[-Infinity,-Infinity,-Infinity];
  for (let i=0; i<count; i++) {
    const offset=header+i*record;
    for (let axis=0; axis<3; axis++) {
      const mm=Math.round(xyz[3*i+axis]*1000);
      if (!Number.isFinite(mm) || Math.abs(mm)>2147483647) throw Error('LASの座標範囲を超えています');
      view.setInt32(offset+4*axis,mm,true);
      min[axis]=Math.min(min[axis],mm); max[axis]=Math.max(max[axis],mm);
      view.setUint16(offset+20+2*axis,colors[3*i+axis]*256,true);
    }
    bytes[offset+14]=9; // First and only return.
  }
  for (let axis=0; axis<3; axis++) {
    view.setFloat64(131+8*axis,.001,true);
    view.setFloat64(179+16*axis,max[axis]/1000,true);
    view.setFloat64(187+16*axis,min[axis]/1000,true);
  }
  return bytes;
}
