import base64
import json
import logging
import shutil
from copy import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import datetime as dt
from pathlib import Path
from typing import List, Literal, Optional

from openpyxl import Workbook
from openpyxl.chart import PieChart, Reference
from pydantic import BaseModel

from . import config, ivit

logger = logging.getLogger(__name__)


@dataclass
class XmlBasicInfo:
    disk_name: str
    keyword: str
    total_num: int
    positive_num: int
    negative_num: int
    rate: float


@dataclass
class XmlSuccessData:
    file_name: str
    detected: str
    result: str


@dataclass
class XmlFailedData:
    file_name: str
    error: str


def image_to_base64(image_path) -> str:
    with open(image_path, "rb") as image_file:
        base64_bytes = base64.b64encode(image_file.read())
        base64_string = base64_bytes.decode("utf-8")
        return f"data:image/png;base64,{base64_string}"


class Report(BaseModel):
    rw_comp: Literal["PASS", "FAIL", None] = None
    status: Literal["PASS", "FAIL"]
    ground_truth: str
    ai_verify: Optional[bool]
    rule_verify: Optional[bool]
    detected: Optional[str]
    confidence: Optional[str | float]
    from_csv: Optional[bool]
    created_time: int
    data: ivit.InferData
    model_info: Optional[ivit.InferModelInfo]
    output_info: config.OUTPUT
    config_info: config.Config


def get_new_file_name(org_path: str) -> str:
    pure_name = Path(org_path).name
    timestamp = dt.now().strftime("%y%m%d%H%M")
    return f"{timestamp}_{pure_name}"


def get_timetamp() -> int:
    return int(dt.now().strftime("%y%m%d%H%M"))


def get_daily_timestamp_for_history() -> int:
    return int(dt.now().strftime("%y%m%d"))


def get_rw_from_domain(domain: str) -> Literal["_R", "_W"]:
    return "_R" if domain == "read" else "_W"


def get_retrain_path(
    retrain_root: str,
    status: bool,
    ground_truth: str,
    domain: str,
) -> Path:
    """
    - retrain
        - <positive/negative> /
            - <R/W> /
                - <groundtruth> /
                    - <data>
    """
    root = Path(retrain_root)
    pn = "postivie" if status else "negative"
    gt = ground_truth.replace('"', " ")
    rw = get_rw_from_domain(domain).replace("_", "")

    dst_dir = root / pn / rw / gt
    return dst_dir


def get_current_path(
    current_root: str,
    status: bool,
    ground_truth: str,
    domain: str,
    timestamp: str,
    data_path: str,
) -> Path:
    """
    - current /
        - <timestamp>_<sn> /
            - <groundtruth>_<positive/negative> /
                - <data>
    """
    root = Path(current_root)
    pn = "postivie" if status else "negative"
    gt = ground_truth.replace('"', " ")
    rw = get_rw_from_domain(domain)

    src_fname = Path(data_path).name
    sn = str(src_fname).split(rw)[0]

    dst_dir = root / f"{timestamp}_{sn}" / f"{gt}_{pn}"
    return dst_dir


def get_history_path(
    history_root: str,
    ground_truth: str,
    domain: str,
    data_path: str,
    timestamp: str,
) -> Path:
    """
    - history /
        - <log>
        - <groundtruth> /
            - <timestamp>_<sn> /
                - <data>
    """
    root = Path(history_root)
    gt = ground_truth.replace('"', " ")
    rw = get_rw_from_domain(domain)

    src_fname = Path(data_path).name
    sn = str(src_fname).split(rw)[0]

    dst_dir = root / gt / f"{timestamp}_{sn}"
    return dst_dir


def get_status_word(status: bool) -> Literal["PASS", "FAIL"]:
    return "PASS" if status else "FAIL"


def get_report(
    infer_data: List[ivit.InferData],
    ground_truth: str,
    model_info: ivit.InferModelInfo,
    config_info: config.Config,
    timestamp: int,
) -> List[Report]:
    reports: List[Report] = []

    all_status = []
    for data in infer_data:
        # 判斷 iVIT 狀態
        ai_verify: Literal[True, False, None] = None
        if config_info.ivit.enable and data.output:
            lower_label = data.output[0].label.lower()
            lower_gt = ground_truth.lower()
            ai_verify = lower_label in lower_gt
        all_status.append(ai_verify)

        # 判斷 rule 狀態
        rule_verify: Literal[True, False, None] = None
        if not config_info.ivit.enable or (
            config_info.ivit.enable and config_info.ivit.rulebase
        ):
            rule_verify = data.input.rule_verify

        # 確認最終狀態
        status: Literal["PASS", "FAIL"] = "FAIL"
        if ai_verify != None and rule_verify != None:
            status = get_status_word(ai_verify and rule_verify)
        elif ai_verify != None and rule_verify == None:
            status = get_status_word(ai_verify)
        elif rule_verify != None and ai_verify == None:
            status = get_status_word(rule_verify)
        logger.warning(
            "Rewrite status: {}, AI: {}, Rule: {}".format(
                status, ai_verify, rule_verify
            )
        )

        # 初始化 Report
        report = Report(
            status=status,
            ground_truth=ground_truth,
            rule_verify=rule_verify,
            ai_verify=ai_verify,
            detected=data.output[0].label,
            confidence=data.output[0].confidence,
            from_csv=config_info.ivit.from_csv,
            data=data,
            model_info=model_info,
            output_info=copy(config_info.output),
            config_info=config_info,
            created_time=timestamp,
        )
        reports.append(report)

    # Update Ground Truth
    for report in reports:
        if not report.model_info:
            continue
        for label in report.model_info.labels:
            dummy_ground_truth = report.ground_truth
            if label in dummy_ground_truth:
                report.ground_truth = label

    # Generatic 模式才有: 判斷 最終的狀態 rw_comp 並更新 path
    config = reports[0].config_info
    if config.ivit.enable and config.ivit.mode == "generatic":
        rw_comp = "FAIL" not in all_status
        for report in reports:
            report.rw_comp = rw_comp

            report.output_info.retrain = str(
                get_retrain_path(
                    retrain_root=config_info.output.retrain,
                    status=report.rw_comp,
                    ground_truth=report.ground_truth,
                    domain=data.input.domain,
                )
            )
            report.output_info.current = str(
                get_current_path(
                    current_root=config_info.output.current,
                    status=report.rw_comp,
                    ground_truth=report.ground_truth,
                    domain=data.input.domain,
                    timestamp=report.created_time,
                    data_path=report.data.input.data_path,
                )
            )
            report.output_info.history = str(
                get_history_path(
                    history_root=config_info.output.history,
                    ground_truth=report.ground_truth,
                    domain=data.input.domain,
                    timestamp=report.created_time,
                    data_path=report.data.input.data_path,
                )
            )

    return reports


def copy_to_retrain(
    reports: List[Report], rw_comp: Literal["PASS", "FAIL", None] = None
):
    """
    Copy files to retrain folder

    Include:
        1. CSV File
        2. Plot file or Screenshot
    """
    for report in reports:
        # Get Correct Dstination (dst)
        dst_dir = Path(report.output_info.retrain)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Select Source File
        src_path = Path(report.data.input.data_path)

        shutil.copy2(src_path, dst_dir)
        if report.config_info.ivit.from_csv:
            for csv_path in src_path.parent.parent.glob("*.csv"):
                shutil.copy2(csv_path, dst_dir)


def copy_to_current(
    reports: List[Report], rw_comp: Literal["PASS", "FAIL", None] = None
):
    """
    Copy files to retrain folder

    Include:
        1. All Files
    """
    for report in reports:
        # Get Correct Dstination (dst)
        dst_dir = Path(report.output_info.current)
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Select Source File
        src_data_path = Path(report.data.input.data_path)
        split_kw = ivit.DOMAIN_KW[report.data.input.domain]
        src_name = str(src_data_path.stem).split(split_kw)[0] + split_kw

        src_dir = Path(report.data.input.data_path).parent
        if report.config_info.ivit.from_csv:
            src_dir = src_dir.parent

        for related_file in src_dir.rglob(f"**/*{src_name}*"):
            shutil.copy2(related_file, dst_dir)


def copy_to_history(
    reports: List[Report], rw_comp: Literal["PASS", "FAIL", None] = None
):
    """
    Copy files to retrain folder

    Include:
        1. All Files
    """
    today = get_daily_timestamp_for_history()
    output_history_dir = reports[0].config_info.output.history
    created_time = reports[0].created_time

    # Re-Write Log for history
    history_log_path = Path(output_history_dir) / f"{today}.log"
    history_log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(history_log_path, "a", encoding="utf-8")

    try:
        log_file.write(
            f"""
[Basic]
Date: {created_time}

[Results]
"""
        )

        for report in reports:
            # Get Correct Dstination (dst)
            dst_dir = Path(report.output_info.history)
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Select Source File
            src_data_path = Path(report.data.input.data_path)
            split_kw = ivit.DOMAIN_KW[report.data.input.domain]
            sn_name = str(src_data_path.stem).split(split_kw)[0]
            src_name = sn_name + split_kw

            src_dir = Path(report.data.input.data_path).parent
            if report.config_info.ivit.from_csv:
                src_dir = src_dir.parent

            for related_file in src_dir.rglob(f"**/*{src_name}*"):
                shutil.copy2(related_file, dst_dir)

            log_file.write(f"""
\t- SN: {sn_name}
\t\t- {split_kw}
\t\t\t- Input Name: {src_data_path.name}
\t\t\t- Input Path: {src_data_path}
\t\t\t- Status: {report.status}
\t\t\t- AI Verify: {report.ai_verify}
\t\t\t- RULE Verify: {report.rule_verify}
\t\t\t- GroundTruth: {report.ground_truth}
\t\t\t- Detected: {report.detected}
\t\t\t- Retrain: {report.output_info.retrain}
\t\t\t- History: {report.output_info.history}
\t\t\t- Current: {report.output_info.current}
\t\t\t- AI Details: {report.data.output}

----------------------------------------------------------------------
""")

    finally:
        log_file.close()


def process(
    reports: List[Report],
):
    for report in reports:
        # Get Top 1 result
        data = report.data

        src_path = Path(data.input.data_path)
        src_dir = src_path.parent
        src_file = src_path.stem

        # Check correct folder and Save Report
        is_ivit_enable = report.config_info.ivit.enable
        if not is_ivit_enable or (is_ivit_enable and report.from_csv):
            src_dir = src_dir.parent

        json_path = src_dir / f"{src_file}.json"
        with open(json_path, "w", encoding="UTF-8") as f:
            json.dump(report.model_dump(), f, ensure_ascii=False, indent=4)
        logger.info(f"Save report to {json_path}")

    cfg = reports[0].config_info
    if not cfg.ivit.enable or (cfg.ivit.enable and cfg.ivit.mode == "generatic"):
        copy_to_retrain(reports=reports)
        copy_to_current(reports=reports)
        copy_to_history(reports=reports)
        logger.info("Copy to target folers")


class DqeXmlHandler:
    XML_EXT: str = ".xlsx"

    def __init__(self, output_dir: str, output_name: Optional[str] = None) -> None:
        if output_name is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{now}{self.XML_EXT}"
        self.xml_dir = Path(output_dir)
        self.xml_dir.mkdir(parents=True, exist_ok=True)

        self.xml_path = self.xml_dir / output_name

        self.wb = Workbook()
        self.first_ws = self.wb.active

    def add_page(self, title: str, contents: List[list]):
        cur_ws = self.wb.create_sheet(title=title)
        for content in contents:
            cur_ws.append(content)
        self.adjust_worksheet(cur_ws=cur_ws)

    def adjust_worksheet(self, cur_ws, scale: float = 1.2, bias: int = 1):
        # 自動調整列寬
        for col in cur_ws.columns:
            max_length = 0
            column = col[0].column_letter  # Get the column name
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except BaseException:
                    pass
            adjusted_width = (max_length + bias) * scale
            cur_ws.column_dimensions[column].width = adjusted_width

    def save_xml(self):
        self.wb.save(self.xml_path)

    def print_xml(self):
        pass


def dc2list(dataclass_instance) -> list:
    return list(asdict(dataclass_instance).values())


def process_xml(reports: List[Report]):
    """處理 Validator 的 XML"""
    gt = reports[0].ground_truth
    kw = get_rw_from_domain(reports[0].data.input.domain).replace("_", "")
    xml_handler = DqeXmlHandler(output_dir="validation")

    wrong_inputs: List[Report] = []
    output_data: List[Report] = []
    for report in reports:
        if report.status == "FAIL":
            wrong_inputs.append(
                XmlFailedData(
                    file_name=Path(report.data.input.data_path).name, error=""
                )
            )
        output_data.append(
            XmlSuccessData(
                file_name=Path(report.data.input.data_path).name,
                detected=report.detected,
                result=report.status,
            )
        )
    # For XML
    num_tot = len(output_data)
    num_pos = len([data for data in output_data if data.result == "PASS"])
    num_neg = num_tot - num_pos
    rate = int((num_pos / num_tot) * 100) if num_tot and num_pos else 0
    basic_info = XmlBasicInfo(
        disk_name=gt,
        keyword=kw,
        total_num=num_tot,
        positive_num=num_pos,
        negative_num=num_neg,
        rate=rate,
    )
    xml_handler.first_ws.title = "Overview"
    xml_handler.first_ws.append(["Category", "Content"])
    for key, val in zip(
        ["Disk", "Mode", "Total", "PASS", "FAIL", "Rate"],
        dc2list(basic_info),
    ):
        xml_handler.first_ws.append([key, val])
    xml_handler.adjust_worksheet(cur_ws=xml_handler.first_ws, bias=4)

    chart = PieChart()
    chart.title = "PASS / FAIL"
    categories = Reference(xml_handler.first_ws, min_col=1, min_row=5, max_row=6)
    data = Reference(xml_handler.first_ws, min_col=2, min_row=5, max_row=6)
    chart.add_data(data, titles_from_data=False)
    chart.set_categories(categories)
    xml_handler.first_ws.add_chart(chart, "E5")

    xml_handler.add_page(
        title="Results",
        contents=[["File Path", "Detected", "Result"]]
        + [dc2list(dc) for dc in output_data],
    )
    xml_handler.add_page(
        title="Failed",
        contents=[["File Path", "Error Message"]]
        + [dc2list(dc) for dc in wrong_inputs],
    )

    xml_handler.save_xml()
