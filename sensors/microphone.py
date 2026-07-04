"""
sensors/microphone.py

Microphone sensor implementation.

Responsibility:
    Capture raw PCM audio chunks from the microphone.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from queue import Queue
from typing import Optional

import numpy as np
import sounddevice as sd

from core.sensor import Sensor
from core.sensor_event import SensorEvent


class MicrophoneSensor(Sensor):
    """
    Captures raw microphone audio.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1600,
        device: Optional[int] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._device = device

        self._queue: Queue[np.ndarray] = Queue()
        self._stream: Optional[sd.InputStream] = None

    @property
    def name(self) -> str:
        return "microphone"

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status,
    ) -> None:
        """
        Called by sounddevice on a background thread.
        """
        if status:
            print(status)

        self._queue.put(indata.copy())

    async def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            blocksize=self._block_size,
            callback=self._callback,
            device=self._device,
            dtype="float32",
        )

        self._stream.start()

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> SensorEvent:
        """
        Wait for the next audio chunk.
        """

        samples = await asyncio.to_thread(
            self._queue.get
        )

        return SensorEvent(
            sensor=self.name,
            timestamp=datetime.now(timezone.utc),
            payload=samples,
        )
    
async def main():
    mic = MicrophoneSensor()

    await mic.start()

    try:
        while True:
            event = await mic.read()

            print(
                event.timestamp,
                event.payload.shape,
                event.payload.dtype,
            )

    finally:
        await mic.stop()


if __name__ == "__main__":
    asyncio.run(main())