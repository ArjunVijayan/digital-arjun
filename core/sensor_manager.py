import asyncio

from core.sensor import Sensor


class SensorManager:

    def __init__(self, sensors: list[Sensor]) -> None:
        self._sensors = sensors

    async def start(self) -> None:

        await asyncio.gather(
            *(sensor.start() for sensor in self._sensors)
        )

    async def stop(self) -> None:

        await asyncio.gather(
            *(sensor.stop() for sensor in self._sensors)
        )

    async def stream(self):

        async with asyncio.TaskGroup() as tg:

            for sensor in self._sensors:

                tg.create_task(self._run_sensor(sensor))

    async def _run_sensor(self, sensor: Sensor):

        while True:

            event = await sensor.read()

            # publish to EventBus later

            print(event)