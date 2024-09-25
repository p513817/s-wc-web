# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
Copyright (C) 2020-2023 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (c) 2023 Innodisk Corporation

 This software is released under the MIT License.

    https://opensource.org/licenses/MIT
"""

import logging as log
import time
from time import perf_counter
from typing import Tuple, Union

import cv2
from numpy import ndarray


def put_highlighted_text(
    frame, message, position, font_face, font_scale, color, thickness
):
    cv2.putText(
        frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1
    )  # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


class Timer:
    def __init__(self) -> None:
        """Simple Timer to calculate execute time"""

        self.pre_time = None
        self.cur_time = perf_counter()
        self.exec_time = None

    def update(self) -> Union[int, float]:
        """Return the execution time between the current and last call

        Returns:
            Union[int, float]: the execution time
        """
        self.pre_time = self.cur_time
        self.cur_time = perf_counter()
        self.exec_time = self.cur_time - self.pre_time
        return self.exec_time

    def get_time(self) -> Union[int, float]:
        """Return the latest execution time

        Returns:
            Union[int, float]: the latest execution time
        """
        return self.exec_time


class Metric:
    def __init__(self, update_times: int = 30) -> None:
        """Performace Metric to get FPS

        Args:
            update_times (int, optional): every N time to update FPS. Defaults to 30.
        """
        self.has_time = False
        self.create_time = perf_counter()
        self.update_times = update_times
        self.reset()

    def reset(self) -> None:
        """Setup parameters"""
        self.pre_time = None
        self.cur_time = None

        self.latencies = 0
        self.cur_update_times = 0
        self.cur_latency = 0

    def update(self) -> Union[int, float]:
        """Return the execution time ( latency ) between the current and last call

        Returns:
            Union[int, float]: current latency

        """
        if self.cur_update_times >= self.update_times:
            self.reset()

        if self.cur_time is None:
            self.cur_time = perf_counter()
            self.has_time = False
            return -1

        self.has_time = True
        self.cur_update_times += 1
        self.pre_time = self.cur_time
        self.cur_time = perf_counter()
        self.cur_latency = self.cur_time - self.pre_time
        self.latencies += self.cur_latency

        return self.cur_latency

    def get_fps(self) -> Union[int, float]:
        """Return Average FPS"""
        return (self.cur_update_times / self.latencies) if self.has_time else -1

    def get_latency(self) -> Union[int, float]:
        """Return Average Latency"""
        return (self.latencies / self.cur_update_times) if self.has_time else -1

    def get_exec_time(self) -> Union[int, float]:
        """Return total execute time"""
        return perf_counter() - self.create_time

    def paint_metrics(
        self,
        frame: ndarray,
        position: Tuple[int, int] = (15, 30),
        font_scale: float = 0.75,
        color: Tuple[int, int, int] = (200, 10, 10),
        thickness: int = 2,
    ):
        """Draw performance stats (Latency, FPS) over frame"""
        if self.get_latency() != -1:
            put_highlighted_text(
                frame,
                f"Latency: {self.get_latency():.5f} ms",
                position,
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color,
                thickness,
            )
        if self.get_fps() != -1:
            put_highlighted_text(
                frame,
                f"FPS: {self.get_fps():.1f}",
                (position[0], position[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale,
                color,
                thickness,
            )

    def log_total(self) -> None:
        """Log out total information"""
        log.info("Metrics report:")
        log.info(
            f"\tLatency: {self.get_latency():.5f} ms"
            if self.get_latency() is not None
            else "\tLatency: N/A"
        )
        log.info(
            f"\tFPS: {self.get_fps():.1f}"
            if self.get_fps() is not None
            else "\tFPS: N/A"
        )


if __name__ == "__main__":
    test = Metric()

    for _ in range(60):
        time.sleep(0.033)
        test.update()

    print(f"Latency: {test.get_latency()}, FPS: {test.get_fps()}")
