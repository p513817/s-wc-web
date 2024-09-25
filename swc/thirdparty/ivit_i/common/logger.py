# -*- coding: UTF-8 -*-

# Copyright (c) 2023 Innodisk Corporation
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Literal, Union

import colorlog

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARN,
    "error": logging.ERROR,
}

SUP_LOG_MODE = ["w", "a", "w+", "a+"]

FIRST_TIME = True


def ivit_logger(
    log_name: Union[str, None] = None,
    write_mode: Literal["w", "a"] = "a",
    level: Literal["debug", "info", "warning", "error"] = "debug",
    clear_log: bool = False,
    log_folder: str = "./logs",
) -> logging.Logger:
    """Initialize iVIT-I Logger

    Args:
        log_name (Union[str, None], optional): it will generate a log file if define it. Defaults to None.
        write_mode (Literal[&#39;w&#39;, &#39;a&#39;], optional): write mode. Defaults to 'a'.
        level (Literal[&#39;debug&#39;, &#39;info&#39;, &#39;warning&#39;, &#39;error&#39;], optional): logger level. Defaults to 'debug'.
        clear_log (bool, optional): if need clear log file. Defaults to False.

    Returns:
        logging.Logger: return logger object
    """

    # Define Global Variable
    global FIRST_TIME

    # Get Default Logger
    logger = logging.getLogger()  # get logger

    # Double Check if logger exist
    if FIRST_TIME and logger.hasHandlers():
        return logger

    # Set Default
    FIRST_TIME = False
    logger.setLevel(logging.DEBUG)  # set default level

    # Add Stream Handler
    formatter = colorlog.ColoredFormatter(
        "%(asctime)s %(log_color)s [%(levelname)-.4s] %(reset)s %(message)s ",
        "%y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(LOG_LEVEL[level])
    logger.addHandler(stream_handler)

    # Return Logger if not setup log_name
    if log_name == "" or log_name is None:
        return logger

    # Add File Handler and clear old one
    if clear_log and os.path.exists(log_name):
        os.remove(log_name)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-.4s] %(message)s (%(filename)s:%(lineno)s)",
        "%y-%m-%d %H:%M:%S",
    )

    # NOTE: before v1.2.1
    # file_handler = logging.FileHandler(os.path.join('/workspace', log_name), write_mode, 'utf-8')

    # Create Logs Folder
    log_folder = os.path.join(".", log_folder)
    if not os.path.exists(log_folder):
        print("Create Logs Folder ...")
        os.makedirs(log_folder)

    # Naming Log file
    create_day = datetime.now().strftime("%y-%m-%d")
    log_name = f"{os.path.splitext(log_name)[0]}-{create_day}.log"

    # Combine Path
    log_path = os.path.join(log_folder, log_name)
    file_handler = RotatingFileHandler(
        log_path,
        mode=write_mode,
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,
        backupCount=1,
        delay=0,
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def dqe_logger(
    log_name: Union[str, None] = "dqe",
    write_mode: Literal["w", "a"] = "a",
    level: Literal["debug", "info", "warning", "error"] = "debug",
    clear_log: bool = False,
    log_folder: str = "./logs",
) -> logging.Logger:
    """Initialize iVIT-I Logger

    Args:
        log_name (Union[str, None], optional): it will generate a log file if define it. Defaults to None.
        write_mode (Literal[&#39;w&#39;, &#39;a&#39;], optional): write mode. Defaults to 'a'.
        level (Literal[&#39;debug&#39;, &#39;info&#39;, &#39;warning&#39;, &#39;error&#39;], optional): logger level. Defaults to 'debug'.
        clear_log (bool, optional): if need clear log file. Defaults to False.

    Returns:
        logging.Logger: return logger object
    """
    # Get Default Logger
    logger = logging.getLogger(log_name)  # get logger

    # Set Default
    logger.setLevel(logging.INFO)  # set default level

    # Add Stream Handler
    # formatter = colorlog.ColoredFormatter( "%(asctime)s [DQE] %(message)s ", "%y-%m-%d %H:%M:%S")
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(LOG_LEVEL[level])
    # logger.addHandler(stream_handler)

    # Return Logger if not setup log_name
    # if log_name == "" or log_name is None:
    #     return logger

    # Add File Handler and clear old one
    if clear_log and os.path.exists(log_name):
        os.remove(log_name)

    # formatter = logging.Formatter( "%(asctime)s %(message)s", "%H:%M:%S")
    formatter = logging.Formatter("%(message)s")

    # NOTE: before v1.2.1
    # file_handler = logging.FileHandler(os.path.join('/workspace', log_name), write_mode, 'utf-8')

    # Create Logs Folder
    check_dir = os.path.dirname(log_folder)
    if not os.path.exists(check_dir):
        raise FileNotFoundError(f"Can not find the folder: {check_dir}")
    log_folder = os.path.join(".", log_folder)
    if not os.path.exists(log_folder):
        print("Create Logs Folder ...")
        os.makedirs(log_folder)

    # Naming Log file
    create_day = datetime.now().strftime("%y-%m-%d")
    log_name = f"{os.path.splitext(log_name)[0]}-{create_day}.log"

    # Combine Path
    log_path = os.path.join(log_folder, log_name)
    file_handler = RotatingFileHandler(
        log_path,
        mode=write_mode,
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,
        backupCount=1,
        delay=0,
    )

    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


ivit_logger(log_name="ivit-i.log", write_mode="a", level="debug", clear_log=False)
