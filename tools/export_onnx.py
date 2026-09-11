from pathlib import Path
from types import MethodType

import onnx
import torch
from onnxsim import simplify
from ultralytics import YOLO

from config import SIZE, WEIGHTS

OPSET_VERSION = 11
IR_VERSION = 7
BATCH_SIZE = 1
NUM_CLASSES = 1
REG_MAX = 16
STRIDES = (8, 16, 32)
OUTPUT_NAMES = (
    "bbox_P3",
    "cls_P3",
    "bbox_P4",
    "cls_P4",
    "bbox_P5",
    "cls_P5",
)


def raw_detect_forward(self, features):
    outputs = []
    for index in range(self.nl):
        outputs.append(self.cv2[index](features[index]))
        outputs.append(self.cv3[index](features[index]))
    return outputs


def export_onnx(weights, image_size):
    model = YOLO(str(weights))
    network = model.model.cpu().eval()
    head = network.model[-1]

    if head.__class__.__name__ != "Detect":
        raise RuntimeError(f"模型检测头不是 Detect: {head.__class__.__name__}")
    if head.nl != len(STRIDES):
        raise RuntimeError(f"模型检测尺度不是 {len(STRIDES)}: {head.nl}")
    if head.nc != NUM_CLASSES or head.reg_max != REG_MAX:
        raise RuntimeError(
            f"检测头参数不符合要求: nc={head.nc}, reg_max={head.reg_max}"
        )

    head.forward = MethodType(raw_detect_forward, head)
    height, width = image_size_shape(image_size)
    dummy = torch.zeros(BATCH_SIZE, 3, height, width)
    output = weights.with_suffix(".onnx")

    with torch.no_grad():
        outputs = network(dummy)
    if len(outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(f"模型输出数量不符合要求: {len(outputs)}")
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        print(f"{name}: {tuple(tensor.shape)}")

    torch.onnx.export(
        network,
        dummy,
        output,
        input_names=["images"],
        output_names=list(OUTPUT_NAMES),
        opset_version=OPSET_VERSION,
        dynamo=False,
        do_constant_folding=True,
    )
    return output


def simplify_onnx(path):
    model = onnx.load(path)
    model, ok = simplify(model)
    if not ok:
        raise RuntimeError("ONNX simplify 校验失败")

    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, path)


def check_onnx(path, image_size):
    model = onnx.load(path)
    opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
    onnx.checker.check_model(model)

    if model.ir_version > IR_VERSION:
        raise RuntimeError(f"ONNX IR version 不符合要求: {model.ir_version}")
    if opsets.get("ai.onnx") not in (10, 11):
        raise RuntimeError(f"ONNX opset 不符合要求: {opsets.get('ai.onnx')}")

    output_names = tuple(output.name for output in model.graph.output)
    if output_names != OUTPUT_NAMES:
        raise RuntimeError(f"ONNX 输出不符合要求: {output_names}")

    height, width = image_size_shape(image_size)
    expected_shapes = []
    for stride in STRIDES:
        output_height = height // stride
        output_width = width // stride
        expected_shapes.append([BATCH_SIZE, 4 * REG_MAX, output_height, output_width])
        expected_shapes.append([BATCH_SIZE, NUM_CLASSES, output_height, output_width])
    output_shapes = [
        [item.dim_value for item in output.type.tensor_type.shape.dim]
        for output in model.graph.output
    ]
    if output_shapes != expected_shapes:
        raise RuntimeError(f"ONNX 输出形状不符合要求: {output_shapes}")

    producers = {name: node for node in model.graph.node for name in node.output}
    output_ops = [producers[name].op_type for name in OUTPUT_NAMES]
    if output_ops != ["Conv"] * len(OUTPUT_NAMES):
        raise RuntimeError(f"ONNX 输出不是检测分支的原始卷积结果: {output_ops}")

    dfl_softmax = [
        node.name
        for node in model.graph.node
        if node.op_type == "Softmax" and "dfl" in node.name.lower()
    ]
    if dfl_softmax:
        raise RuntimeError(f"ONNX 仍包含 DFL Softmax: {dfl_softmax}")

    print(f"ONNX: {path}")
    print(f"input: batch={BATCH_SIZE}, size={image_size_text(image_size)}")
    for output, shape in zip(model.graph.output, output_shapes):
        print(f"output: {output.name} {shape}")
    print(f"opset: {opsets.get('ai.onnx')}")
    print(f"IR version: {model.ir_version}")


def image_size_text(image_size):
    if isinstance(image_size, int):
        return f"{image_size}x{image_size}"
    return f"{image_size[0]}x{image_size[1]}"


def image_size_shape(image_size):
    if isinstance(image_size, int):
        return image_size, image_size
    return image_size


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"模型不存在: {WEIGHTS}")

    output = export_onnx(WEIGHTS, SIZE)
    simplify_onnx(output)
    check_onnx(output, SIZE)


if __name__ == "__main__":
    main()
