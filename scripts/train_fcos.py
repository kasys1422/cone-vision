"""Fine-tune an FCOS FPN detector and export its dense ONNX head."""
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
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.detection import FCOS, FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FrozenBatchNorm2d, box_iou, nms
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

SIZE = 512


def read_split(directory, archive, split, *, synthetic=False):
    records = []
    prefix = "synthetic" if synthetic else "real_detection"
    manifest = directory / f"{prefix}_{split}.jsonl"
    if synthetic:
        suffix = f"/images/{split}/000000.jpg"
        roots = [name[:-len(suffix)] for name in archive.namelist() if name.endswith(suffix)]
        if len(roots) != 1:
            raise ValueError(f"Expected one synthetic image root containing {suffix}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if synthetic:
            filename = Path(record["image"].replace("\\", "/")).name
            member = f"{roots[0]}/images/{split}/{filename}"
        else:
            member = "Cone-Detection-main/TraCon_dataset/" + record["image_relative"]
        data = archive.read(member)
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


def mixed_epoch(real, synthetic, epoch, sampler):
    # Keep the real-only run's image/update budget. Round across epochs so
    # that one in three sampled images comes from the synthetic train split.
    n = len(real)
    synthetic_count = ((epoch + 1) * n) // 3 - (epoch * n) // 3
    batch = sampler.sample(real, n - synthetic_count) + sampler.sample(synthetic, synthetic_count)
    sampler.shuffle(batch)
    return batch


def build_model(backbone="resnet50"):
    if backbone == "resnet18":
        body = resnet18(weights=ResNet18_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        for name, parameter in body.named_parameters():
            parameter.requires_grad_(name.startswith(("layer3.", "layer4.")))
        pyramid = BackboneWithFPN(
            body, return_layers={"layer2": "0", "layer3": "1", "layer4": "2"},
            in_channels_list=[128, 256, 512], out_channels=128,
            extra_blocks=LastLevelP6P7(512, 128),
        )
        return FCOS(pyramid, num_classes=1, min_size=SIZE, max_size=SIZE)

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
    parser.add_argument("--backbone", choices=("resnet50", "resnet18"), default="resnet50")
    parser.add_argument("--synthetic-zip", type=Path)
    parser.add_argument("--mixed-epochs", type=int, default=0,
                        help="Use real:synthetic 2:1 for these epochs, then finish on real images")
    args = parser.parse_args()
    if not 0 <= args.mixed_epochs < args.epochs:
        parser.error("--mixed-epochs must leave at least one real-only epoch")
    if bool(args.synthetic_zip) != bool(args.mixed_epochs):
        parser.error("Supply --synthetic-zip and --mixed-epochs together")
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13)
    random.seed(13)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00012)
    with zipfile.ZipFile(args.tracon_zip) as archive:
        train = read_split(args.dataset,archive,"train")
        val = read_split(args.dataset,archive,"val")
    print("Training / validation images",len(train),len(val),flush=True)
    sampler = random.Random(13)
    if args.mixed_epochs:
        with zipfile.ZipFile(args.synthetic_zip) as archive:
            synthetic = read_split(args.dataset, archive, "train", synthetic=True)
        print("Synthetic training images", len(synthetic), flush=True)
    history = []
    best_f1 = -1
    for epoch in range(args.epochs):
        model.train()
        if epoch < args.mixed_epochs:
            epoch_images = mixed_epoch(train, synthetic, epoch, sampler)
        else:
            random.shuffle(train)
            epoch_images = train
        for start in range(0,len(epoch_images),4):
            images,targets = [],[]
            for _,image,gt,(scale,pad,_) in epoch_images[start:start+4]:
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
        if epoch >= args.mixed_epochs and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
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
