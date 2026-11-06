"""Convert official DA3 weights to a single-image, depth-only, 504px Q4 ONNX."""

import argparse
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from unittest.mock import patch

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from safetensors import safe_open
import torch
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dualdpt import DualDPT


WEIGHT_DIGESTS = {
    "DA3-BASE": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
    "DA3-LARGE-1.1": "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64",
}


class SingleImageDepth(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = DinoV2(**{k: v for k, v in config["net"].items() if k != "__object__"})
        self.head = DualDPT(**{k: v for k, v in config["head"].items() if k != "__object__"})

    def forward(self, image):
        features, _ = self.backbone(image.unsqueeze(1))
        return self.head(features, image.shape[-2], image.shape[-1], patch_start_idx=0).depth


def load_model(directory):
    weights = directory / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHT_DIGESTS[directory.name]:
        raise ValueError("Checkpoint does not match the official release")
    model = SingleImageDepth(json.loads((directory / "config.json").read_text())["config"])
    with safe_open(weights, framework="pt") as tensors:
        state = {key.removeprefix("model."): tensors.get_tensor(key) for key in tensors.keys()
                 if key.startswith(("model.backbone.", "model.head."))}
        # Some checkpoints omit unused convolutions in the auxiliary camera-ray head.
        missing, extra = model.load_state_dict(state, strict=False)
        if extra or any(not key.startswith("head.scratch.output_conv2_aux.") for key in missing):
            raise ValueError(f"Unexpected checkpoint keys: missing={missing}, extra={extra}")
    return model.eval()


def export(directory, output):
    if output.exists():
        raise FileExistsError(output)
    model = load_model(directory)
    # ONNX has no cartesian_prod operator. Preserve upstream RoPE's (y,x) order.
    def grid_positions(y, x):
        return torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).reshape(-1, 2)

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "depth.onnx"
        with torch.inference_mode(), patch.object(torch, "cartesian_prod", grid_positions):
            torch.onnx.export(model, torch.zeros(1, 3, 504, 504), str(fp32), dynamo=False,
                              opset_version=17, input_names=["image"], output_names=["depth"])
        del model
        quantizer = MatMulNBitsQuantizer(onnx.load(fp32), block_size=32, is_symmetric=True)
        quantizer.process()
        graph = quantizer.model.model
        if not any(node.op_type == "MatMulNBits" for node in graph.graph.node):
            raise ValueError("No linear weights were quantized")
        onnx.checker.check_model(graph)
        onnx.save(graph, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="DA3-BASE or DA3-LARGE-1.1 with config and weights")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(6)
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    export(args.directory, args.output)
