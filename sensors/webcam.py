"""
sensors/webcam.py

Webcam sensor implementation.

Responsibility:
    Capture frames from the default webcam and expose them as SensorEvents.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import cv2

from core.sensor import Sensor
from core.sensor_event import SensorEvent


class WebcamSensor(Sensor):
    """
    Webcam sensor.

    Captures frames from a webcam at a configurable FPS.
    """

    def __init__(
        self,
        camera_index: int = 0,
        fps: float = 15.0,
    ) -> None:
        self._camera_index = camera_index
        self._fps = fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_interval = 1.0 / fps

    @property
    def name(self) -> str:
        return "webcam"

    async def start(self) -> None:
        """
        Open the webcam.
        """
        self._capture = cv2.VideoCapture(self._camera_index)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open webcam {self._camera_index}"
            )

    async def stop(self) -> None:
        """
        Release webcam resources.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    async def read(self) -> SensorEvent:
        """
        Capture a single frame.

        Returns
        -------
        SensorEvent
            Sensor event containing the captured frame.
        """
        if self._capture is None:
            raise RuntimeError(
                "WebcamSensor.start() must be called before read()."
            )

        success, frame = await asyncio.to_thread(
            self._capture.read
        )

        if not success:
            raise RuntimeError("Failed to capture webcam frame.")

        await asyncio.sleep(self._frame_interval)

        return SensorEvent(
            sensor=self.name,
            timestamp=datetime.now(timezone.utc),
            payload=frame,
        )
    
if __name__ == "__main__":
    async def main():
        webcam_sensor = WebcamSensor(fps=5.0)
        await webcam_sensor.start()

        try:
            for _ in range(10):
                event = await webcam_sensor.read()
                print(f"Captured frame at {event.timestamp}")
                cv2.imshow("Webcam Frame", event.payload)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            await webcam_sensor.stop()

    asyncio.run(main())