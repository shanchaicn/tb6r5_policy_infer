"""Intel RealSense camera interface for policy inference (RGB only)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pyrealsense2 as rs

from .base_camera import BaseCameraInterface


class RealSenseCameraInterface(BaseCameraInterface):
    """Poll one or more RealSense cameras by serial number."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_numbers: list[str] | None = None,
        enable_depth: bool = False,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_numbers = serial_numbers
        self.enable_depth = enable_depth
        self.pipelines: dict[str, rs.pipeline] = {}
        self.configs: dict[str, rs.config] = {}
        self.align: dict[str, rs.align] = {}
        self.frames_dict: dict[str, dict] = {}
        self.frames_lock = threading.Lock()
        self.last_update_time: dict[str, float] = {}

        context = rs.context()
        devices = context.query_devices()
        if not devices:
            raise RuntimeError("No Intel RealSense devices connected.")

        device_serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if self.serial_numbers:
            self.active_serials = [s for s in self.serial_numbers if s in device_serials]
            if not self.active_serials:
                raise RuntimeError(f"Specified RealSense devices with serials {self.serial_numbers} not found.")
        else:
            self.active_serials = device_serials

        for serial in self.active_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.pipelines[serial] = pipeline
            self.configs[serial] = config
            if self.enable_depth:
                self.align[serial] = rs.align(rs.stream.color)
            self.last_update_time[serial] = 0.0
            print(f"Initialized RealSense camera: {serial}")

    def start(self) -> None:
        for serial, pipeline in self.pipelines.items():
            pipeline.start(self.configs[serial])
            print(f"Started pipeline for camera: {serial}")

    def update_frames(self) -> None:
        current_time = time.time()
        frames_dict: dict[str, dict] = {}

        for serial, pipeline in self.pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
                color_frame = None
                depth_frame = None
                if self.enable_depth:
                    aligned = self.align[serial].process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                else:
                    color_frame = frames.get_color_frame()

                if not color_frame:
                    print(f"Warning: No color frame available from camera {serial}")
                    continue

                frames_dict[serial] = {
                    "color": np.asanyarray(color_frame.get_data()).copy(),
                    "depth": np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None,
                    "timestamp_us": color_frame.get_timestamp(),
                }
                self.last_update_time[serial] = current_time
            except RuntimeError as exc:
                if "timeout" in str(exc).lower():
                    print(
                        f"Frame timeout for camera {serial} "
                        f"(last successful: {current_time - self.last_update_time[serial]:.2f}s ago)"
                    )
                else:
                    print(f"Error getting frames from {serial}: {exc}")

        with self.frames_lock:
            self.frames_dict = frames_dict

    def get_frames(self) -> dict:
        with self.frames_lock:
            return self.frames_dict.copy()

    def get_frame(self, serial: str):
        with self.frames_lock:
            return self.frames_dict[serial].copy() if serial in self.frames_dict else None

    def stop(self) -> None:
        for serial, pipeline in self.pipelines.items():
            try:
                pipeline.stop()
                print(f"Stopped pipeline for camera: {serial}")
            except RuntimeError as exc:
                print(f"Error stopping pipeline for {serial}: {exc}")
