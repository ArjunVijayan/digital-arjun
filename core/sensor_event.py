from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class SensorEvent:

    sensor: str

    timestamp: datetime

    payload: Any