"""Camera streaming for TB6-R5 policy inference (RealSense, V4L2, or HTTP URL)."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from .constants import INFER_LOG_PREFIX


class CameraStreamProtocol(Protocol):
    def start(self) -> None: ...

    def wait_ready(self, timeout_s: float = 10.0) -> None: ...

    def get_images(self) -> dict[str, np.ndarray]: ...

    def stop(self) -> None: ...


def to_rgb_hwc_uint8(color: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize to RGB HWC uint8; resize if needed."""
    arr = np.asarray(color)
    if arr.ndim == 3 and (arr.shape[0] != height or arr.shape[1] != width):
        import cv2

        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


_PREVIEW_WINDOWS: set[str] = set()


def show_camera_rgb(images: dict[str, np.ndarray]) -> None:
    """Show RGB frames in OpenCV windows (RGB -> BGR for imshow)."""
    import cv2

    for name, rgb in images.items():
        if rgb is None:
            continue
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        window = f"{INFER_LOG_PREFIX} RGB - {name}"
        if window not in _PREVIEW_WINDOWS:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            _PREVIEW_WINDOWS.add(window)
        cv2.imshow(window, bgr)
    cv2.waitKey(1)


def destroy_camera_windows() -> None:
    import cv2

    _PREVIEW_WINDOWS.clear()
    cv2.destroyAllWindows()


class CameraPreview:
    """Refresh OpenCV preview on a dedicated thread (decoupled from inference loop)."""

    def __init__(self, cam_stream: CameraStreamProtocol, fps: float = 30.0):
        self._cam_stream = cam_stream
        self._dt = 1.0 / max(fps, 1.0)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            start_t = time.time()
            imgs = self._cam_stream.get_images()
            if imgs:
                show_camera_rgb(imgs)
            elapsed = time.time() - start_t
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _parse_name_value_pairs(spec: str, *, option: str, example: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        if not name or not value:
            raise ValueError(f"Invalid {option} entry: '{pair}' (expected {example})")
        out[name.strip()] = value.strip()
    if not out:
        raise ValueError(f"{option} must list at least one name=value pair")
    return out


def parse_camera_serials(spec: str | None, default_serial_dict: dict[str, str]) -> dict[str, str]:
    if not spec:
        return dict(default_serial_dict)
    return _parse_name_value_pairs(spec, option="--camera-serials", example="name=serial")


def parse_camera_devices(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    return _parse_name_value_pairs(
        spec,
        option="--camera-devices",
        example="name=/dev/video0 or name=0",
    )


def parse_camera_urls(spec: str | None) -> dict[str, str]:
    if not spec:
        return {}
    return _parse_name_value_pairs(
        spec,
        option="--camera-urls",
        example="name=http://host:8888/RsCameraSensor/0/0/color",
    )


def create_camera_stream(
    *,
    camera_urls: str | None,
    camera_devices: str | None,
    camera_serials: str | None,
    default_serial_dict: dict[str, str],
    width: int,
    height: int,
    fps: int,
) -> tuple[CameraStreamProtocol | None, list[str]]:
    """Open HTTP URL, V4L2, or RealSense streams. Returns (stream, sorted observation names)."""
    url_dict = parse_camera_urls(camera_urls)
    if url_dict:
        if camera_serials or camera_devices:
            print(f"[{INFER_LOG_PREFIX}][camera] --camera-urls set; ignoring --camera-serials and --camera-devices")
        names = sorted(url_dict.keys())
        stream = HttpCameraStream(url_dict, width, height, fps)
        return stream, names

    device_dict = parse_camera_devices(camera_devices)
    if device_dict:
        if camera_serials:
            print(f"[{INFER_LOG_PREFIX}][camera] --camera-devices set; ignoring --camera-serials")
        names = sorted(device_dict.keys())
        stream = V4l2CameraStream(device_dict, width, height, fps)
        return stream, names

    serial_dict = parse_camera_serials(camera_serials, default_serial_dict)
    names = sorted(serial_dict.keys())
    stream = RealSenseCameraStream(serial_dict, width, height, fps)
    return stream, names


class RealSenseCameraStream:
    """Owns a RealSenseCameraInterface plus a background polling thread."""

    def __init__(self, serial_dict: dict[str, str], width: int, height: int, fps: int):
        from .hardware.realsense import RealSenseCameraInterface

        self.serial_dict = serial_dict
        self.serial_to_name = {serial: name for name, serial in serial_dict.items()}
        self.width = width
        self.height = height
        self.cam = RealSenseCameraInterface(
            width=width,
            height=height,
            fps=fps,
            serial_numbers=list(serial_dict.values()),
            enable_depth=False,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.cam.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cam.update_frames()
            except Exception as exc:
                print(f"[{INFER_LOG_PREFIX}][camera] update_frames error: {exc}")
                time.sleep(0.02)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.serial_dict.values())
        while time.time() < deadline:
            frames = self.cam.get_frames()
            if needed.issubset(set(frames.keys())):
                print(f"[{INFER_LOG_PREFIX}][camera] all RealSense cameras streaming")
                return
            time.sleep(0.1)
        print(f"[{INFER_LOG_PREFIX}][camera] WARNING: not all RealSense cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        frames = self.cam.get_frames()
        out: dict[str, np.ndarray] = {}
        for serial, name in self.serial_to_name.items():
            fd = frames.get(serial)
            if fd is not None and fd.get("color") is not None:
                out[name] = to_rgb_hwc_uint8(fd["color"], self.height, self.width)
        return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.cam.stop()
        except Exception:
            pass


class V4l2CameraStream:
    """Capture RGB via OpenCV VideoCapture (/dev/video* or numeric index)."""

    def __init__(self, device_dict: dict[str, str], width: int, height: int, fps: int):
        self.device_dict = device_dict
        self.width = width
        self.height = height
        self.fps = fps
        self._caps: dict[str, object] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _open_capture(device: str):
        import cv2

        if device.isdigit():
            cap = cv2.VideoCapture(int(device), cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        return cap

    def start(self) -> None:
        import cv2

        for name, device in self.device_dict.items():
            cap = self._open_capture(device)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open V4L2 camera '{name}' ({device})")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._caps[name] = cap
            print(f"[{INFER_LOG_PREFIX}][camera] opened V4L2 '{name}' -> {device}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import cv2

        while not self._stop.is_set():
            for name, cap in self._caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._frames[name] = to_rgb_hwc_uint8(rgb, self.height, self.width)
            time.sleep(max(0.0, 1.0 / max(self.fps, 1) * 0.5))

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.device_dict.keys())
        while time.time() < deadline:
            with self._lock:
                ready = needed.issubset(set(self._frames.keys()))
            if ready:
                print(f"[{INFER_LOG_PREFIX}][camera] all V4L2 cameras streaming")
                return
            time.sleep(0.1)
        print(f"[{INFER_LOG_PREFIX}][camera] WARNING: not all V4L2 cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: frame.copy() for name, frame in self._frames.items()}

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        for name, cap in self._caps.items():
            try:
                cap.release()
            except Exception as exc:
                print(f"[{INFER_LOG_PREFIX}][camera] release '{name}' error: {exc}")
        self._caps.clear()


class HttpCameraStream:
    """Poll JPEG (or raw image bytes) from HTTP GET endpoints per camera."""

    def __init__(
        self,
        url_dict: dict[str, str],
        width: int,
        height: int,
        fps: int,
        *,
        timeout_s: float = 10.0,
    ):
        self.url_dict = url_dict
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_s = timeout_s
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    @staticmethod
    def _extract_jpeg_bytes(data: bytes) -> bytes | None:
        """Support raw JPEG or multipart/x-mixed-replace (e.g. RsCameraSensor --frame)."""
        if not data:
            return None
        start = data.find(b"\xff\xd8")
        if start < 0:
            return None
        end = data.find(b"\xff\xd9", start)
        if end < 0:
            return None
        return data[start : end + 2]

    @staticmethod
    def _decode_image_bytes(data: bytes) -> np.ndarray | None:
        import cv2

        jpeg = HttpCameraStream._extract_jpeg_bytes(data)
        if jpeg is None:
            return None
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _camera_loop(self, name: str, url: str) -> None:
        """Keep one HTTP connection open and consume MJPEG frames as they arrive."""
        import urllib.error
        import urllib.request

        while not self._stop.is_set():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "tb6r5-policy-infer/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    buf = b""
                    while not self._stop.is_set():
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            jpeg = self._extract_jpeg_bytes(buf)
                            if jpeg is None:
                                break
                            end = buf.find(b"\xff\xd9", buf.find(b"\xff\xd8")) + 2
                            buf = buf[end:]
                            rgb = self._decode_image_bytes(jpeg)
                            if rgb is not None:
                                with self._lock:
                                    self._frames[name] = to_rgb_hwc_uint8(rgb, self.height, self.width)
                        if len(buf) > 2_000_000:
                            buf = buf[-200_000:]
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if not self._stop.is_set():
                    print(f"[{INFER_LOG_PREFIX}][camera] HTTP stream '{name}' disconnected: {exc}")
                time.sleep(0.3)

    def start(self) -> None:
        for name, url in self.url_dict.items():
            print(f"[{INFER_LOG_PREFIX}][camera] opened HTTP '{name}' -> {url}")
            thread = threading.Thread(target=self._camera_loop, args=(name, url), daemon=True)
            thread.start()
            self._threads.append(thread)

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        needed = set(self.url_dict.keys())
        while time.time() < deadline:
            with self._lock:
                ready = needed.issubset(set(self._frames.keys()))
            if ready:
                print(f"[{INFER_LOG_PREFIX}][camera] all HTTP cameras streaming")
                return
            time.sleep(0.1)
        print(f"[{INFER_LOG_PREFIX}][camera] WARNING: not all HTTP cameras produced frames before timeout")

    def get_images(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {name: frame.copy() for name, frame in self._frames.items()}

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()


# Backward-compatible alias used before V4L2 support.
CameraStream = RealSenseCameraStream
