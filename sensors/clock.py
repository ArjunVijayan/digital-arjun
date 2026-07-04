"""
sensors/clock.py

Clock sensor implementation.

Responsibility:
    Generate periodic clock tick events.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Final

from core.sensor import Sensor
from core.sensor_event import SensorEvent


class ClockSensor(Sensor):
    """
    Periodic clock sensor.

    Emits a SensorEvent at a configurable interval.
    """

    DEFAULT_INTERVAL: Final[float] = 1.0

    def __init__(self, interval: float = DEFAULT_INTERVAL) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero.")

        self._interval = interval

    @property
    def name(self) -> str:
        return "clock"

    @property
    def interval(self) -> float:
        return self._interval

    async def start(self) -> None:
        """
        Nothing to initialize.
        """
        return

    async def stop(self) -> None:
        """
        Nothing to clean up.
        """
        return

    async def read(self) -> SensorEvent:
        """
        Wait until the next clock tick.
        """

        await asyncio.sleep(self._interval)

        now = datetime.now(timezone.utc)

        return SensorEvent(
            sensor=self.name,
            timestamp=now,
            payload={
                "utc": now,
                "unix": now.timestamp(),
            },
        )


if __name__ == "__main__":

    async def main() -> None:
        clock = ClockSensor(interval=1)

        await clock.start()

        try:
            while True:
                event = await clock.read()
                print(event)

        except KeyboardInterrupt:
            pass

        finally:
            await clock.stop()

    asyncio.run(main())