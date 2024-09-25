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
"""

from time import perf_counter
from typing import Dict, Set


def parse_devices(device_string):
    colon_position = device_string.find(":")
    if colon_position != -1:
        device_type = device_string[:colon_position]
        if device_type == "HETERO" or device_type == "MULTI":
            comma_separated_devices = device_string[colon_position + 1 :]
            devices = comma_separated_devices.split(",")
            for device in devices:
                parenthesis_position = device.find(":")
                if parenthesis_position != -1:
                    device = device[:parenthesis_position]
            return devices
    return (device_string,)


def parse_value_per_device(devices: Set[str], values_string: str) -> Dict[str, int]:
    """Format: <device1>:<value1>,<device2>:<value2> or just <value>"""
    values_string_upper = values_string.upper()
    result = {}
    device_value_strings = values_string_upper.split(",")
    for device_value_string in device_value_strings:
        device_value_list = device_value_string.split(":")
        if len(device_value_list) == 2:
            if device_value_list[0] in devices:
                result[device_value_list[0]] = int(device_value_list[1])
        elif len(device_value_list) == 1 and device_value_list[0] != "":
            for device in devices:
                result[device] = int(device_value_list[0])
        elif device_value_list[0] != "":
            raise RuntimeError(f"Unknown string format: {values_string}")
    return result


def get_user_config(
    flags_d: str, flags_nstreams: str, flags_nthreads: int
) -> Dict[str, str]:
    config = {}

    devices = set(parse_devices(flags_d))

    device_nstreams = parse_value_per_device(devices, flags_nstreams)
    for device in devices:
        if device == "CPU":  # CPU supports a few special performance-oriented keys
            # limit threading for CPU portion of inference
            if flags_nthreads:
                config["CPU_THREADS_NUM"] = str(flags_nthreads)

            config["CPU_BIND_THREAD"] = "NO"

            # for CPU execution, more throughput-oriented execution via streams
            config["CPU_THROUGHPUT_STREAMS"] = (
                str(device_nstreams[device])
                if device in device_nstreams
                else "CPU_THROUGHPUT_AUTO"
            )
        elif device == "GPU":
            config["GPU_THROUGHPUT_STREAMS"] = (
                str(device_nstreams[device])
                if device in device_nstreams
                else "GPU_THROUGHPUT_AUTO"
            )
            if "MULTI" in flags_d and "CPU" in devices:
                # multi-device execution with the CPU + GPU performs best with GPU throttling hint,
                # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                config["GPU_PLUGIN_THROTTLE"] = "1"
    return config


class SyncPipeline:
    def __init__(self, model):
        self.model = model
        self.model.load()
        self.completed_results = {}

        self.callback_exceptions = []
        self.result = None

    def submit_data(self, frame):
        try:
            inputs, preprocess_meta = self.model.preprocess(frame)
            result = None
            raw_result = self.model.infer_sync(inputs)
            if raw_result:
                meta = {"frame": frame, "start_time": perf_counter()}
                result = (
                    self.model.postprocess(raw_result, preprocess_meta),
                    {**meta, **preprocess_meta},
                )

            return result

        except Exception as e:
            self.callback_exceptions.append(e)

    def get_result(self):
        return self.result

    def is_ready(self):
        return True

    def await_all(self):
        pass

    def await_any(self):
        pass
