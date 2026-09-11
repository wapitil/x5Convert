from pathlib import Path

import cv2
import numpy as np
from config import CONF, DATA, ROOT
from prepare_calibration import prepare_image

IOU_THRESHOLD = 0.7
MATCH_IOU = 0.5
REG_MAX = 16
STRIDES = (8, 16, 32)


def create_session(path):
    from horizon_tc_ui import HB_ONNXRuntime

    session = HB_ONNXRuntime(model_file=str(path))
    return session, session.input_names[0], session.output_names


def infer(session_info, tensor):
    session, input_name, output_names = session_info
    inputs = {input_name: tensor[np.newaxis]}
    return session.run(output_names, inputs)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def decode_outputs(outputs):
    boxes = []
    scores = []
    bins = np.arange(REG_MAX, dtype=np.float32)

    for level, stride in enumerate(STRIDES):
        bbox = outputs[level * 2][0]
        cls = outputs[level * 2 + 1][0]
        _, height, width = bbox.shape

        distances = bbox.reshape(4, REG_MAX, height, width)
        distances = distances - distances.max(axis=1, keepdims=True)
        distances = np.exp(distances)
        distances /= distances.sum(axis=1, keepdims=True)
        distances = (distances * bins[None, :, None, None]).sum(axis=1)

        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32) + 0.5,
            np.arange(width, dtype=np.float32) + 0.5,
            indexing="ij",
        )
        left = (grid_x - distances[0]) * stride
        top = (grid_y - distances[1]) * stride
        right = (grid_x + distances[2]) * stride
        bottom = (grid_y + distances[3]) * stride
        boxes.append(np.stack((left, top, right, bottom), axis=-1).reshape(-1, 4))
        scores.append(sigmoid(cls).transpose(1, 2, 0).reshape(-1, cls.shape[0]))

    return np.concatenate(boxes), np.concatenate(scores)


def postprocess(outputs):
    boxes, scores = decode_outputs(outputs)
    class_scores = scores.max(axis=1)
    selected = class_scores >= CONF
    boxes = boxes[selected]
    class_scores = class_scores[selected]
    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    nms_boxes = boxes.copy()
    nms_boxes[:, 2] -= nms_boxes[:, 0]
    nms_boxes[:, 3] -= nms_boxes[:, 1]
    keep = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), class_scores.tolist(), CONF, IOU_THRESHOLD
    )
    keep = np.asarray(keep).reshape(-1)

    return np.column_stack((boxes[keep], class_scores[keep])).astype(np.float32)


def output_error(reference, candidate):
    bbox_error = np.mean(
        [np.mean(np.abs(reference[index] - candidate[index])) for index in (0, 2, 4)]
    )
    score_error = np.mean(
        [np.mean(np.abs(reference[index] - candidate[index])) for index in (1, 3, 5)]
    )
    return float(bbox_error), float(score_error)


def box_iou(box, boxes):
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-9)


def compare_detections(reference, candidate):
    available = list(range(len(candidate)))
    matches = []
    for detection in reference:
        if not available:
            break
        candidates = candidate[available]
        ious = box_iou(detection[:4], candidates[:, :4])
        best = int(np.argmax(ious))
        if ious[best] >= MATCH_IOU:
            index = available.pop(best)
            matches.append(
                (float(ious[best]), abs(float(detection[4] - candidate[index, 4])))
            )
    return matches


def main():
    model_dir = ROOT / "model_output/accuracy"
    model_paths = {
        "source": ROOT / "model/best.onnx",
        "original_float": model_dir / "best_original_float_model.onnx",
        "optimized_float": model_dir / "best_optimized_float_model.onnx",
        "calibrated": model_dir / "best_calibrated_model.onnx",
        "quantized": model_dir / "best_quantized_model.onnx",
    }
    test_dir = DATA / "images/test"

    for path in model_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"模型不存在: {path}")

    sessions = {name: create_session(path) for name, path in model_paths.items()}
    paths = sorted(
        path
        for path in test_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )

    stats = {
        name: {"count": 0, "matches": [], "box_error": [], "score_error": []}
        for name in model_paths
        if name != "source"
    }
    source_count = 0
    for path in paths:
        tensor = prepare_image(path)
        outputs = {name: infer(session, tensor) for name, session in sessions.items()}
        source_boxes = postprocess(outputs["source"])
        source_count += len(source_boxes)

        counts = [f"source={len(source_boxes)}"]
        for name, values in stats.items():
            boxes = postprocess(outputs[name])
            matches = compare_detections(source_boxes, boxes)
            values["count"] += len(boxes)
            values["matches"].extend(matches)
            box_error, score_error = output_error(outputs["source"], outputs[name])
            values["box_error"].append(box_error)
            values["score_error"].append(score_error)
            counts.append(f"{name}={len(boxes)}/{len(matches)}")
        print(f"{path.name}: " + ", ".join(counts))

    print(f"测试图片: {len(paths)}")
    print(f"source 检测框: {source_count}")
    for name, values in stats.items():
        matches = values["matches"]
        mean_iou = np.mean([item[0] for item in matches]) if matches else 0.0
        mean_score_diff = np.mean([item[1] for item in matches]) if matches else 0.0
        print(f"\n{name}")
        print(f"检测框: {values['count']}, 匹配框: {len(matches)}")
        print(f"匹配框平均 IoU: {mean_iou:.6f}")
        print(f"匹配框平均置信度差: {mean_score_diff:.6f}")
        print(f"原始坐标通道平均绝对误差: {np.mean(values['box_error']):.6f}")
        print(f"原始置信度通道平均绝对误差: {np.mean(values['score_error']):.6f}")


if __name__ == "__main__":
    main()
