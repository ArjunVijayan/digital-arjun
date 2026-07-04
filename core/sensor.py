from __future__ import annotations

from abc import ABC, abstractmethod

from core.sensor_event import SensorEvent


class Sensor(ABC):
    """
    Base class for every sensor in the system.

    A sensor has exactly one responsibility:
    continuously produce SensorEvents.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique sensor name."""

    @abstractmethod
    async def start(self) -> None:
        """Allocate resources."""

    @abstractmethod
    async def stop(self) -> None:
        """Release resources."""

    @abstractmethod
    async def read(self) -> SensorEvent:
        """
        Read a single observation.

        Never blocks forever.
        """