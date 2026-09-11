import cv2
import numpy as np

from config import DATA, EXTS, ROOT, SIZE, TOP

CALIBRATION_COUNT = 100
VIDEO_SAMPLES_PER_FILE = 10
PADDING_VALUE = 114


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full((size, size, 3), PADDING_VALUE, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def prepare_frame(image):
    image = letterbox(image, SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def prepare_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return prepare_frame(image)


def save_tensor(tensor, target, expected_bytes):
    tensor.tofile(target)
    if target.stat().st_size != expected_bytes:
        raise RuntimeError(f"标定文件大小错误: {target}")


def prepare_videos(output, expected_bytes):
    video_dir = ROOT.parent / "test"
    count = 0
    for path in sorted(video_dir.glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indexes = np.linspace(
            frame_count * 0.1,
            frame_count * 0.9,
            VIDEO_SAMPLES_PER_FILE,
            dtype=int,
        )

        for index, frame_index in enumerate(frame_indexes, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"无法读取视频帧: {path}, frame={frame_index}")

            crop_top = int(frame.shape[0] * TOP)
            tensor = prepare_frame(frame[crop_top:, :])
            target = output / f"video_{path.stem}_{index:02d}.bin"
            save_tensor(tensor, target, expected_bytes)
            count += 1

        capture.release()
    return count


def main():
    output = ROOT / "calibration_data_f32"
    output.mkdir(parents=True, exist_ok=True)

    paths = []
    for split in ("train", "val"):
        source = DATA / "images" / split
        paths.extend(
            path for path in sorted(source.iterdir()) if path.suffix.lower() in EXTS
        )
    paths = paths[:CALIBRATION_COUNT]
    if not paths:
        raise RuntimeError(f"没有找到标定图片: {DATA / 'images'}")

    expected_bytes = 3 * SIZE * SIZE * np.dtype(np.float32).itemsize
    for path in paths:
        tensor = prepare_image(path)
        target = output / f"{path.stem}.bin"
        save_tensor(tensor, target, expected_bytes)

    video_count = prepare_videos(output, expected_bytes)

    print(f"标定数据: {output}")
    print(f"图片样本: {len(paths)}")
    print(f"视频样本: {video_count}")
    print(f"样本总数: {len(paths) + video_count}")
    print(f"单个文件: {expected_bytes} bytes, RGB NCHW float32 [0, 1]")


if __name__ == "__main__":
    main()
