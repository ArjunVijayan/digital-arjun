"""
sensors/screen.py

Screen sensor implementation.

Responsibility:
    Capture screenshots from the primary display.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import mss
import cv2

import numpy as np

from core.sensor import Sensor
from core.sensor_event import SensorEvent


class ScreenSensor(Sensor):
    """
    Captures screenshots from the primary monitor.
    """

    def __init__(
        self,
        monitor: int = 1,
        fps: float = 2.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        self._monitor = monitor
        self._fps = fps
        self._frame_interval = 1.0 / fps

        self._mss: Optional[mss.mss] = None
        self._monitor_info: Optional[dict] = None

    @property
    def name(self) -> str:
        return "screen"

    async def start(self) -> None:
        """
        Initialize screen capture.
        """
        self._mss = mss.mss()

        try:
            self._monitor_info = self._mss.monitors[self._monitor]
        except IndexError as exc:
            raise RuntimeError(
                f"Monitor {self._monitor} does not exist."
            ) from exc

    async def stop(self) -> None:
        """
        Release resources.
        """
        if self._mss is not None:
            self._mss.close()
            self._mss = None

    async def read(self) -> SensorEvent:
        """
        Capture one screenshot.
        """
        if self._mss is None or self._monitor_info is None:
            raise RuntimeError(
                "ScreenSensor.start() must be called before read()."
            )

        screenshot = await asyncio.to_thread(
            self._mss.grab,
            self._monitor_info,
        )

        frame = np.asarray(screenshot)[..., :3]

        await asyncio.sleep(self._frame_interval)

        return SensorEvent(
            sensor=self.name,
            timestamp=datetime.now(timezone.utc),
            payload=frame,
        )


if __name__ == "__main__":

    async def main() -> None:
        sensor = ScreenSensor(fps=1)

        await sensor.start()

        try:
            while True:
                event = await sensor.read()

                print(
                    event.timestamp,
                    event.payload.shape,
                    event.payload.dtype,
                )

                cv2.imshow("Screen Capture", event.payload)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            await sensor.stop()

    asyncio.run(main())