"""Fine-tune one FCOS ResNet-50 FPN detector and export its dense ONNX head."""
import argparse
import hashlib
import io
import json
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.ops import box_iou, nms

SIZE = 512


def read_split(directory, archive, split):
    records = []
    for line in (directory / f"real_detection_{split}.jsonl").read_text().splitlines():
        record = json.loads(line)
        data = archive.read("Cone-Detection-main/TraCon_dataset/" + record["image_relative"])
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Image hash mismatch: {record['image_id']}")
        image = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"))
        h, w = image.shape[:2]
        if (w, h) != (record["width"], record["height"]):
            raise ValueError(record["image_id"])
        scale = SIZE/max(h,w)
        nw, nh = max(1,round(w*scale)), max(1,round(h*scale))
        x,y = (SIZE-nw)//2,(SIZE-nh)//2
        canvas = np.full((SIZE,SIZE,3),114,np.uint8)
        canvas[y:y+nh,x:x+nw] = cv2.resize(image,(nw,nh))
        boxes = torch.tensor([o["bbox_xyxy"] for o in record["objects"]],dtype=torch.float32).reshape(-1,4)
        transform = torch.tensor([nw/w,nh/h,nw/w,nh/h]), torch.tensor([x,y,x,y]), (w,h)
        records.append((record["image_id"], torch.from_numpy(canvas).permute(2,0,1), boxes, transform))
    return records


def build_model():
    model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
                              min_size=SIZE, max_size=SIZE, trainable_backbone_layers=2)
    head = model.head.classification_head
    head.num_classes = 1
    head.cls_logits = nn.Conv2d(256,1,3,padding=1)
    nn.init.normal_(head.cls_logits.weight,std=.01)
    nn.init.constant_(head.cls_logits.bias,-4.595)
    return model


class DenseFCOS(nn.Module):
    """Fixed 512px letterbox input; emit pixel cx,cy,w,h,score,1 before NMS."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head
        self.coder = model.head.box_coder
        image = torch.zeros(3,SIZE,SIZE,device=next(model.parameters()).device)
        images,_ = model.transform([image])
        feats = list(self.backbone(images.tensors).values())
        self.register_buffer("anchors",model.anchor_generator(images,feats)[0])
        self.register_buffer("mean",image.new_tensor(model.transform.image_mean).view(1,3,1,1))
        self.register_buffer("std",image.new_tensor(model.transform.image_std).view(1,3,1,1))

    def forward(self, images):
        outputs = self.head(list(self.backbone((images-self.mean)/self.std).values()))
        boxes = self.coder.decode(outputs["bbox_regression"][0],self.anchors)
        center = (boxes[:,:2]+boxes[:,2:])/2
        extent = boxes[:,2:]-boxes[:,:2]
        score = (outputs["cls_logits"][0].sigmoid()*outputs["bbox_ctrness"][0].sigmoid()).sqrt()
        return torch.cat([center,extent,score,torch.ones_like(score)],dim=1).unsqueeze(0)


@torch.no_grad()
def evaluate(model, records, device):
    dense = DenseFCOS(model.eval()).eval()
    counts = dict(tp=0,fp=0,fn=0,small_tp=0,small_n=0,empty_fp=0)
    rows = {}
    for image_id, image, gt, (scale,pad,(w,h)) in records:
        raw = dense(image[None].to(device).float()/255)[0].cpu()
        order = raw[:,4].argsort(descending=True)[:1000]
        raw = raw[order[raw[order,4] >= .50]]
        boxes = (torch.cat([raw[:,:2]-raw[:,2:4]/2,raw[:,:2]+raw[:,2:4]/2],1)-pad)/scale
        # Clip to the original frame exactly as the Web decoder does.
        boxes[:,::2].clamp_(0,w)
        boxes[:,1::2].clamp_(0,h)
        keep = ((boxes[:,2:]-boxes[:,:2])>1).all(dim=1)
        boxes,scores = boxes[keep],raw[keep,4]
        idx = nms(boxes,scores,.5)[:100]
        boxes,scores = boxes[idx],scores[idx]
        rows[image_id] = [{"box":b.tolist(),"score":s.item()} for b,s in zip(boxes,scores)]
        used = set()
        small = gt[:,3]-gt[:,1]<96
        counts["small_n"] += int(small.sum())
        for overlaps in box_iou(boxes,gt):
            overlaps[list(used)] = -1
            if len(gt) and overlaps.max() >= .5:
                i = int(overlaps.argmax())
                used.add(i)
                counts["small_tp"] += int(small[i])
            else:
                counts["fp"] += 1
                counts["empty_fp"] += int(not len(gt))
        counts["tp"] += len(used)
        counts["fn"] += len(gt)-len(used)
    counts["f1"] = 2*counts["tp"]/(2*counts["tp"]+counts["fp"]+counts["fn"])
    return counts,rows


def pack_for_web(source, target):
    # Compact storage only. Cast weights back to float32 before convolution so
    # WASM does not require an FP16 arithmetic provider.
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(source)
    casts = []
    for tensor in model.graph.initializer:
        if tensor.data_type != TensorProto.FLOAT or len(tensor.raw_data) < 4096:
            continue
        name = tensor.name
        values = numpy_helper.to_array(tensor).astype(np.float16)
        tensor.CopyFrom(numpy_helper.from_array(values, name + "_packed"))
        casts.append(helper.make_node("Cast", [tensor.name], [name], to=TensorProto.FLOAT))
    original = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(casts + original)
    onnx.checker.check_model(model)
    onnx.save(model, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=Path)
    parser.add_argument("tracon_zip",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--epochs",type=int,default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13)
    random.seed(13)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00012)
    with zipfile.ZipFile(args.tracon_zip) as archive:
        train = read_split(args.dataset,archive,"train")
        val = read_split(args.dataset,archive,"val")
    print("Training / validation images",len(train),len(val),flush=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train)
        for start in range(0,len(train),4):
            images,targets = [],[]
            for _,image,gt,(scale,pad,_) in train[start:start+4]:
                image = image.to(device).float()/255
                boxes = (gt*scale+pad).to(device)
                if random.random()<.5:
                    image = image.flip(-1)
                    boxes[:,[0,2]] = SIZE-boxes[:,[2,0]]
                image = (image*random.uniform(.8,1.2)+random.uniform(-.06,.06)).clamp(0,1)
                targets.append({"boxes":boxes,"labels":torch.zeros(len(gt),dtype=torch.long,device=device)})
                images.append(image)
            losses = model(images,targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step()
        metrics,_ = evaluate(model,val,device)
        print(epoch+1,metrics,flush=True)
        if not history or metrics["f1"] > max(row["f1"] for row in history):
            torch.save(model.state_dict(),args.output/"best.pt")
        history.append(metrics)
        if epoch+1 == args.epochs//2:
            for group in optimizer.param_groups:
                group["lr"] *= .35
    model.load_state_dict(torch.load(args.output/"best.pt",map_location=device,weights_only=True))
    with zipfile.ZipFile(args.tracon_zip) as archive:
        test = read_split(args.dataset,archive,"test")
    metrics,rows = evaluate(model,test,device)
    print("TEST",metrics,flush=True)
    (args.output/"test_predictions.json").write_text(json.dumps(rows))
    (args.output/"metrics.json").write_text(json.dumps({"validation":history,"test":metrics},indent=2))
    dense = DenseFCOS(model.eval().cpu()).eval()
    torch.onnx.export(dense,torch.zeros(1,3,SIZE,SIZE),args.output/"cone_fcos.onnx",
                      input_names=["images"],output_names=["predictions"],opset_version=17,dynamo=False)
    pack_for_web(args.output/"cone_fcos.onnx", args.output/"cone_detector.onnx")


if __name__ == "__main__":
    main()
