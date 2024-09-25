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

import numpy as np
from numpy import ndarray
import logging as log
from time import perf_counter
from typing import Union, Literal, Optional, List
from .types import NumericalValue, ListValue, StringValue
from .utils import softmax

from .image_model import ImageModel
from .model import iModel

from ..pipelines import get_user_config, AsyncPipeline, SyncPipeline
from ..adapters import create_core, OpenvinoAdapter

class Classification(ImageModel):
    __model__ = 'Classification'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self._check_io_number(1, 1)
        if self.path_to_labels:
            self.labels = self._load_labels(self.path_to_labels)
        self.out_layer_name = self._get_outputs()

        # Check label length
        self.topk = len(self.labels) if len(self.labels) <= self.topk else self.topk

    def _load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = []
            for line in f.readlines():
                labels.append(line.strip())
        return labels

    def _get_outputs(self):
        layer_name = next(iter(self.outputs))
        layer_shape = self.outputs[layer_name].shape

        if len(layer_shape) != 2 and len(layer_shape) != 4:
            self.raise_error('The Classification model wrapper supports topologies only with 2D or 4D output')
        if len(layer_shape) == 4 and (layer_shape[2] != 1 or layer_shape[3] != 1):
            self.raise_error('The Classification model wrapper supports topologies only with 4D '
                             'output which has last two dimensions of size 1')
        if self.labels:
            if (layer_shape[1] == len(self.labels) + 1):
                self.labels.insert(0, 'other')
                self.logger.warning("\tInserted 'other' label as first.")
            if layer_shape[1] != len(self.labels):
                self.raise_error("Model's number of classes and parsed "
                                 'labels must match ({} != {})'.format(layer_shape[1], len(self.labels)))
        return layer_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['resize_type'].update_default_value('standard')
        parameters.update({
            'topk': NumericalValue(value_type=int, default_value=1, min=1),
            'labels': ListValue(description="List of class labels"),
            'path_to_labels': StringValue(
                description="Path to file with labels. Overrides the labels, if they sets via 'labels' parameter"
            ),
            'confidence_threshold': NumericalValue(value_type=float, default_value=0.1, min=0),
        })
        return parameters

    def postprocess(self, outputs, meta):
        outputs = outputs[self.out_layer_name].squeeze()

        indices = np.argpartition(outputs, -self.topk)[-self.topk:]
        
        # Get the score
        scores = outputs[indices]

        # Sort score from big to small
        desc_order = scores.argsort()[::-1]
        
        scores = scores[desc_order]
        indices = indices[desc_order]

        if not np.isclose(np.sum(outputs), 1.0, atol=0.01):
            scores = softmax(scores)
        
        # Add Label
        # labels = [self.labels[i] if self.labels else "" for i in indices]
        
        # Threshold
        new_indices, new_scores, new_labels = [], [], []
        for idx, scr in zip(indices, scores):
            if scr > self.confidence_threshold:
                new_indices.append(idx)
                new_scores.append(scr)
                new_labels.append(self.labels[idx])
        
        # Add Bounding Box Information
        # xmin = [0 for _ in indices]
        # ymin = [0 for _ in indices]
        # xmax = [0 for _ in indices]
        # ymax = [0 for _ in indices]
        
        # return list(zip(indices, labels, scores, xmin, ymin, xmax, ymax))

        return list(zip(new_indices, new_labels, new_scores))


class iClassification(iModel):
    
    def __init__(   self, 
                    model_path:str, 
                    label_path:str, 
                    device:Literal['CPU', 'GPU', None]='CPU', 
                    topk: int= 3,
                    mean_values:Optional[List[float]]=None,
                    scale_values:Optional[List[float]]=None,
                    reverse_input_channels:Optional[bool]=False,
                    confidence_threshold:Union[float, int]=0.1,
                    **kwargs ) -> None:
        """iVIT Classification

        Args:
            - model_path (str): path to model file ( xml )
            - label_path (str): path to label file
            - device (Literal[&#39;CPU&#39;, &#39;GPU&#39;, None], optional): device type. Defaults to 'CPU'.
            - topk (int, optional): get k outputs. Defaults to 3.
            - confidence_threshold (Union[float, int], optional): define the threshold of the confidence. Defaults to 0.9.
            - mean_values (Optional[List[float]], optional): custom mean value. Defaults to None.
            - scale_values (Optional[List[float]], optional): custom scale value. Defaults to None.
            - reverse_input_channels (Optional[bool], optional): if need reverse channels. Defaults to False.
        """
        
        device = 'CPU' if (device is None) else device
        super().__init__(model_path, label_path, device, **kwargs)

        # Wrapper Config for Intel Adaptor
        configuration = {
            'mean_values':  mean_values,
            'scale_values': scale_values,
            'reverse_input_channels': reverse_input_channels,
            'topk': topk,
            'path_to_labels': label_path,
            'confidence_threshold': confidence_threshold,
        }

        # Load Model Wrapper
        # `self.model_adapter` will be instance after super().__init__()
        self.model = Classification(self.model_adapter, configuration)
        self.model.load()
        self.model.log_layers_info()

    def inference(self, frame: ndarray) -> list:
        return super().inference(frame)