import json
import logging
from pathlib import Path
from typing import List, Literal, Optional

import cv2
from pydantic import BaseModel

from swc import thirdparty
from swc.handlers import config

logger = logging.getLogger(__name__)

KW_R = "_R"
KW_W = "_W"

DOMAIN_KW = {"read": KW_R, "write": KW_W}


class InferInput(BaseModel):
    data_path: str | Path
    plot_path: str | Path
    verify_path: str | Path
    domain: Literal["read", "write"]
    rule_verify: Optional[bool] = None


class InferOutput(BaseModel):
    index: Optional[int] = None
    label: Optional[str] = None
    confidence: Optional[float] = None


class InferData(BaseModel):
    input: InferInput
    output: List[InferOutput]


class InferModelInfo(BaseModel):
    xml_path: str
    label_path: str
    config_path: str

    name: str
    platform: str
    arch: str
    classes: int
    labels: list
    input_shape: list

    device: str = "CPU"
    threshold: float = 0.1


def parse_ivit_model_dir(model_dir: str) -> InferModelInfo:
    """return (xml_file_path, cfg_file_path, label_file_path)"""
    model_dir: Path = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Can not find iVIT Model Folder: {model_dir}")

    xml_files, cfg_files, label_files = [], [], []
    for file in model_dir.iterdir():
        if file.suffix == ".xml":
            xml_files.append(file)
        elif file.suffix == ".json":
            cfg_files.append(file)
        elif file.suffix == ".txt":
            label_files.append(file)

    for trg_files in (xml_files, cfg_files, label_files):
        if len(trg_files) != 1:
            raise RuntimeError(
                "Parse file error, please ensure the iVIT Model Folder has Model ( .xml ), Label (.txt), Config (.json)"
            )

    return get_model_info(xml_files[0], cfg_files[0], label_files[0])


def validate(cfg: config.Config):
    if not cfg.ivit.enable:
        return

    is_validator = cfg.ivit.mode == "validator"

    if not cfg.aida.enable and not cfg.ivit.input_dir:
        raise RuntimeError("Input image / csv folder must be settup")

    if is_validator and not cfg.ivit.target_model:
        raise RuntimeError("Must choose at least one target model")

    for model in (cfg.ivit.models.read, cfg.ivit.models.write):
        if is_validator and model != cfg.ivit.target_model:
            continue

        parse_ivit_model_dir(model.model_dir)

        if not (0.0 < float(model.thres) < 1.0):
            raise RuntimeError("Threshold must less than 1.0 and larger than 0")


def get_model_info(xml_path: str, config_path: str, label_path: str) -> InferModelInfo:
    with open(label_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    with open(config_path, "r") as f:
        config = json.load(f)

    return InferModelInfo(
        xml_path=str(xml_path),
        label_path=str(label_path),
        config_path=str(config_path),
        name=str(Path(xml_path).parent),
        labels=labels,
        platform=config["export_platform"],
        arch=config["model_config"]["arch"],
        classes=config["model_config"]["classes"],
        input_shape=config["model_config"]["input_shape"],
    )


def get_model(
    _info: InferModelInfo,
) -> thirdparty.ivit_i.core.models.iClassification:
    return thirdparty.ivit_i.core.models.iClassification(
        model_path=_info.xml_path,
        label_path=_info.label_path,
        confidence_threshold=_info.threshold,
        device=_info.device,
    )


def do_inference(
    model, infer_data_list: List[InferInput], from_csv: bool = True
) -> List[InferData]:
    ret = []
    for infer_data in infer_data_list:
        frame = cv2.imread(str(infer_data.data_path))
        if not from_csv:
            frame = thirdparty.process.shot_to_plot.process.process(frame)
        results = model.inference(frame)
        if results:
            infer_results = [
                InferOutput(
                    index=index,
                    label=label,
                    confidence=conf,
                )
                for (index, label, conf) in results
            ]
        else:
            infer_results = [InferOutput()]
        ret.append(InferData(input=infer_data, output=infer_results))
    return ret
