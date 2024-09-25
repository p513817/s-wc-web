#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)


class BaseModelWrap(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class SSD(BaseModelWrap):
    mode: Literal["detect", "mock"] = "detect"
    mock_name: Optional[str] = None
    ismart_path: Optional[str] = None


class EnableModel(BaseModelWrap):
    enable: bool = True


class AIDA(EnableModel):
    """AIDA 會產生資料夾到 out_dir 然後需要帶入關鍵字 dir_kw"""

    exec_path: Optional[str] = None
    out_dir: Optional[str] = None
    dir_kw: Optional[str] = "aida64v730"


class IVITModel(BaseModelWrap):
    model_config = ConfigDict(protected_namespaces=())
    model_dir: Optional[str] = None
    thres: float = 0.1


class IVIT_RW_Model(BaseModelWrap):
    read: IVITModel = IVITModel()
    write: IVITModel = IVITModel()


class IVIT(EnableModel):
    mode: Optional[Literal["generatic", "validator"]] = "generatic"
    target_model: Optional[Literal["read", "write"]] = None
    from_csv: Optional[bool] = False
    rulebase: Optional[bool] = True
    input_dir: Optional[str] = None
    models: Optional[IVIT_RW_Model] = IVIT_RW_Model()


class OUTPUT(BaseModelWrap):
    out_dir: Optional[str] = None
    retrain: str = "retrain"
    current: str = "current"
    history: str = "history"

    @model_validator(mode="before")
    def update_paths(cls, values):
        out_dir = values.get("out_dir")
        if out_dir:
            # 如果 out_dir 不是 None，修改路徑
            values["retrain"] = str(Path(out_dir) / values.get("retrain", "retrain"))
            values["current"] = str(Path(out_dir) / values.get("current", "current"))
            values["history"] = str(Path(out_dir) / values.get("history", "history"))
        return values


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    ssd: SSD = SSD()
    aida: AIDA = AIDA()
    ivit: IVIT = IVIT()
    output: OUTPUT = OUTPUT()
    debug: bool = False


def get(setting_path: str = "config.json") -> Config:
    if not Path(setting_path).exists():
        logger.info("Get default config")
        return Config()

    with open(setting_path, "r") as f:
        data = json.load(f)
    logger.info(f"Get config from {setting_path}")
    return Config(**data)


def save(setting_dict: dict, setting_path: str = "config.json") -> Config:
    with open(setting_path, "w") as f:
        json.dump(setting_dict, f, ensure_ascii=False, indent=4)
    logger.info(f"Save config to {setting_path}")
