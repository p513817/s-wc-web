import logging
import time
from pathlib import Path

from swc.thirdparty import ismart

from . import config

logger = logging.getLogger(__name__)


class SSDError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def validation(cfg: config.Config):
    if cfg.ssd.mode == "mock":
        if cfg.ssd.mock_name in ["", None]:
            raise SSDError("SSD Mock Name is empty")
        logger.info(f"Validated mock mode: {cfg.ssd.mock_name}")
    else:
        ismart_path = Path(cfg.ssd.ismart_path)
        if ismart_path.suffix != ".exe":
            raise SSDError("Setup iSMART executable file error, should be *.exe")
        if not ismart_path.exists():
            raise SSDError("iSMART executable file not exists")


def valid_detected_disks(disks: list):
    if len(disks) == 0:
        raise SSDError("Can not detect any devices")
    logger.info(f"Validated detect mode: {disks}")


def detect(ismart_path: str) -> list:
    logger.warning("Detect ssd with general mode")

    test_products = ismart.wmic.get_test_product()

    return (
        [
            ismart.wmic.get_name_from_ismart(product, ismart_exec_path=ismart_path)
            for product in test_products
        ]
        if test_products
        else []
    )


def mock_detect(ismart_path: str = "") -> list:
    logger.warning("Detect ssd with mock mode")
    time.sleep(1)
    return ["4TG2-P", "3TE6"]
