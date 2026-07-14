from abc import ABC, abstractmethod


class BaseCameraInterface(ABC):
    """Abstract base class for camera interfaces."""

    @abstractmethod
    def start(self) -> None:
        """Start the camera stream."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the camera stream."""

    @abstractmethod
    def update_frames(self) -> None:
        """Fetch new frames (polling interfaces)."""

    @abstractmethod
    def get_frames(self) -> dict:
        """Return frames keyed by camera identifier."""

    @abstractmethod
    def get_frame(self, identifier: str):
        """Return frames for one camera."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
