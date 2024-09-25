# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
 Copyright (c) 2021-2023 Intel Corporation

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

import logging as log
from numpy import ndarray
from typing import Literal, Union, Optional, List

from .types import ListValue, NumericalValue, StringValue
from .image_model import ImageModel

from .utils import load_labels, clip_detections, RESIZE_TYPES
from .utils import DetectionWithLandmarks, InputTransform, OutputTransform, RESIZE_TYPES

from .image_model import ImageModel
from .model import iModel

from ..pipelines import get_user_config, AsyncPipeline, SyncPipeline
from ..adapters import create_core, OpenvinoAdapter

class DetectionModel(ImageModel):
    '''An abstract wrapper for object detection model

    The DetectionModel must have a single image input.
    It inherits `preprocess` from `ImageModel` wrapper. Also, it defines `_resize_detections` method,
    which should be used in `postprocess`, to clip bounding boxes and resize ones to original image shape.

    The `postprocess` method must be implemented in a specific inherited wrapper.
    '''

    def __init__(self, model_adapter, configuration=None, preload=False):
        '''Detection Model constructor

        It extends the `ImageModel` construtor.

        Args:
            model_adapter (ModelAdapter): allows working with the specified executor
            configuration (dict, optional): it contains values for parameters accepted by specific
              wrapper (`confidence_threshold`, `labels` etc.) which are set as data attributes
            preload (bool, optional): a flag whether the model is loaded to device while
              initialization. If `preload=False`, the model must be loaded via `load` method before inference

        Raises:
            WrapperError: if the model has more than 1 image inputs
        '''

        super().__init__(model_adapter, configuration, preload)

        if not self.image_blob_name:
            self.raise_error("The Wrapper supports only one image input, but {} found".format(
                len(self.image_blob_names)))

        if self.path_to_labels:
            self.labels = load_labels(self.path_to_labels)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'confidence_threshold': NumericalValue(default_value=0.5, description="Threshold value for detection box confidence"),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            )
        })

        return parameters

    def _resize_detections(self, detections, meta):
        '''Resizes detection bounding boxes according to initial image shape.

        It implements image resizing depending on the set `resize_type`(see `ImageModel` for details).
        Next, it applies bounding boxes clipping.

        Args:
            detections (List[Detection]): list of detections with coordinates in normalized form
            meta (dict): the input metadata obtained from `preprocess` method

        Returns:
            - list of detections with resized and clipped coordinates fit to initial image

        Raises:
            WrapperError: If the model uses custom resize or `resize_type` is not set
        '''
        resized_shape = meta['resized_shape']
        original_shape = meta['original_shape']

        if self.resize_type == 'fit_to_window_letterbox':
            detections = resize_detections_letterbox(detections, original_shape[1::-1], resized_shape[1::-1])
        elif self.resize_type == 'fit_to_window':
            detections = resize_detections_with_aspect_ratio(detections, original_shape[1::-1], resized_shape[1::-1], (self.w, self.h))
        elif self.resize_type == 'standard':
            detections = resize_detections(detections, original_shape[1::-1])
        else:
            self.raise_error('Unknown resize type {}'.format(self.resize_type))
        return clip_detections(detections, original_shape)

def resize_detections(detections, original_image_size):
    for detection in detections:
        detection.xmin *= original_image_size[0]
        detection.xmax *= original_image_size[0]
        detection.ymin *= original_image_size[1]
        detection.ymax *= original_image_size[1]
    return detections

def resize_detections_with_aspect_ratio(detections, original_image_size, resized_image_size, model_input_size):
    scale_x = model_input_size[0] / resized_image_size[0] * original_image_size[0]
    scale_y = model_input_size[1] / resized_image_size[1] * original_image_size[1]
    for detection in detections:
        detection.xmin *= scale_x
        detection.xmax *= scale_x
        detection.ymin *= scale_y
        detection.ymax *= scale_y
    return detections

def resize_detections_letterbox(detections, original_image_size, resized_image_size):
    scales = [x / y for x, y in zip(resized_image_size, original_image_size)]
    scale = min(scales)
    scales = (scale / scales[0], scale / scales[1])
    offset = [0.5 * (1 - x) for x in scales]
    for detection in detections:
        detection.xmin = ((detection.xmin - offset[0]) / scales[0]) * original_image_size[0]
        detection.xmax = ((detection.xmax - offset[0]) / scales[0]) * original_image_size[0]
        detection.ymin = ((detection.ymin - offset[1]) / scales[1]) * original_image_size[1]
        detection.ymax = ((detection.ymax - offset[1]) / scales[1]) * original_image_size[1]
    return detections

class iDetection(iModel):

    def __init__(   self, 
                    model_path:str, 
                    label_path:str, 
                    architecture_type:Literal['yolo', 'yolov4'], 
                    device:Literal['CPU', 'GPU', None]='CPU',
                    confidence_threshold:Union[float, int]=0.9,
                    anchors: Optional[str] = None,
                    resize_type: Optional[str] = None, 
                    mean_values:Optional[List[float]]=None,
                    scale_values:Optional[List[float]]=None,
                    reverse_input_channels:Optional[bool]=False,
                    **kwargs ) -> None:
        """iVIT Object Detection

        Args:
            - model_path (str): path to model file ( xml )
            - label_path (str): path to label file
            - device (Literal[&#39;CPU&#39;, &#39;GPU&#39;, None], optional): device type. Defaults to 'CPU'.
            - architecture_type (str): architecture type
            - confidence_threshold (Union[float, int], optional): define the threshold of the confidence. Defaults to 0.9.
            - anchors (Optional[str], optional): _description_. Defaults to None.
            - resize_type (Optional[str], optional): resize type for openvino function. Defaults to None.
            - mean_values (Optional[List[float]], optional): custom mean value. Defaults to None.
            - scale_values (Optional[List[float]], optional): custom scale value. Defaults to None.
            - reverse_input_channels (Optional[bool], optional): if need reverse channels. Defaults to False.
        """
        device = 'CPU' if (device is None) else device
        super().__init__(model_path, label_path, device, **kwargs)

        # Wrapper Config
        configuration = {
            'resize_type': resize_type,
            'mean_values':  mean_values,
            'scale_values': scale_values,
            'reverse_input_channels': reverse_input_channels,
            'path_to_labels': label_path,
            'confidence_threshold': confidence_threshold,
            'anchors': anchors,
        }
    
        # Load Model Wrapper
        self.model = DetectionModel.create_model(architecture_type, self.model_adapter, configuration)
        self.model.load()
        self.model.log_layers_info()
        
    @classmethod
    def available_wrappers(self):
        return DetectionModel.available_wrappers()

    def inference(self, frame: ndarray) -> list:
        return super().inference(frame)