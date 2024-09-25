import logging
import subprocess as sp
from pathlib import Path
from typing import List, Tuple, Literal, Optional
from . import ivit, config
from swc import thirdparty

logger = logging.getLogger(__name__)


class AIDAError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


def mock_run_cmd(output_dir: str = ""):
    logger.info("Run mock AIDA")
    # 使用 Popen 來捕獲 stdout
    process = sp.Popen(
        [
            "python",
            r"C:\Users\max_chang\Desktop\s-wc-web\testing\fake_aida\fake_aida.py",
            "-o",
            output_dir,
        ],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        text=True,  # 將輸出解碼為字串
    )

    # 持續讀取 stdout
    for line in process.stdout:
        print(line, end="")  # 直接輸出每行

    # 確保程序結束後捕獲 stderr（如有）
    process.wait()
    stderr_output = process.stderr.read()
    if stderr_output:
        raise AIDAError(f"Error: {stderr_output}")


def run_cmd(aida_exec_path: str):
    logger.info("Run AIDA")
    is_aida64_empty = False
    process = sp.Popen(
        aida_exec_path,
        shell=True,
        encoding="utf-8",
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        universal_newlines=True,
    )
    while True:
        line = process.stdout.readline().strip().replace("\n", "")
        if line:
            if "[]" in line:
                is_aida64_empty = True
                break
        if line == "" and process.poll() is not None:
            break
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise AIDAError(f"Run AIDA64 Failed: {process.stderr}")
    if is_aida64_empty:
        raise AIDAError("Run AIDA64 Failed: Can not find any testing disk")

    return None


def validate(cfg: config.Config):
    if not cfg.aida.enable:
        logger.warning("AIDA is disable")
        return

    if not Path(cfg.aida.exec_path).exists():
        raise AIDAError("Can not find the executable file of AIDA")

    if not Path(cfg.aida.out_dir).is_dir():
        raise AIDAError("The output folder is not exists or not a directory")

    if cfg.aida.dir_kw in ["", None]:
        raise AIDAError("The output folder keyword is empty")

    folders = get_output_dir(cfg.aida.out_dir, cfg.aida.dir_kw)
    if len(folders) != 0:
        raise AIDAError("The AIDA output folder already exists")


def get_output_dir(folder: str, keyword: str) -> List[Path]:
    return list(Path(folder).glob(f"*{keyword}*"))


def valid_aida_output_folder(gen_folders: List[Path]) -> None:
    """Get a list of folder generated from AIDA"""
    if len(gen_folders) == 1:
        return
    raise AIDAError("The AIDA output folder is not generated")


def parse_aida_output_dir(folder: str, is_validator: bool=False) -> Tuple[List[Path]]:
    """Parse images and csvs from AIDA Output Folder"""
    images, csvs = [], []
    for file in Path(folder).iterdir():
        if file.suffix in [".png", ".jpg"]:
            images.append(file)
        elif file.suffix in [".csv"]:
            csvs.append(file)

    if not is_validator and len(images) != 2:
        raise AIDAError("The AIDA folder images must be 2")
    if not is_validator and len(csvs) != 2:
        raise AIDAError("The AIDA folder csv must be 2")

    return images, csvs



def get_data_domain(data) -> Tuple[Literal["Linear Write", "Linear Read"], Literal["read", "write"]]:
    if ivit.KW_R in str(data):
        return "Linear Read", "read"
    elif ivit.KW_W in str(data):
        return "Linear Write", "write"
    else:
        raise RuntimeError("Can not find keyword in data path")

def get_data(
    cfg: config.Config,
) -> Tuple[List[ivit.InferInput], List[ivit.InferInput]]:

    # Get Mode
    is_csv_mode = not cfg.ivit.enable or (cfg.ivit.enable and cfg.ivit.from_csv)
    logger.warning(f"Is CSV Mode: {is_csv_mode} ( iVIT: {cfg.ivit.enable}, iVIT From CSV: {cfg.ivit.from_csv})")

    # Get Correct Folder
    input_dir: Path | str = cfg.ivit.input_dir
    if cfg.aida.enable:
        logger.info("Detect AIDA enable while getting data")
        input_dirs = get_output_dir(cfg.aida.out_dir, cfg.aida.dir_kw)
        valid_aida_output_folder(gen_folders=input_dirs)
        input_dir = input_dirs[0]
        logger.info(f"Find input folder: {input_dir}")
        
    # Get Correct Data
    images, csvs = parse_aida_output_dir(input_dir, cfg.ivit.mode=="validator")

    # Get correct data and data type
    input_data_list: List[Path] = images
    if is_csv_mode:
        input_data_list = csvs

    # Verify Data
    if cfg.ivit.enable:
        logging.info("Verify data ... Mode: {}, Image: {}, CSV: {}".format(cfg.ivit.mode, len(images), len(csvs)))
        if cfg.ivit.mode == "generatic" and len(input_data_list) != 2:
            raise RuntimeError("Generatic Mode only support 2 images or 2 csvs")
        elif cfg.ivit.mode == "validator" and len(input_data_list) == 0:
            raise RuntimeError(f"Get empty data in {cfg.ivit.input_dir}")

    # Process data
    read_data, write_data = [], []
    output_plot_folder = Path(input_dir) / "plots"
    output_plot_folder.mkdir(parents=True, exist_ok=True)
    for data in input_data_list:
        
        # Get data
        plot_keyword, domain = get_data_domain(data)
        plot_path = data_path = verify_path = str(data)
        rulebased_status = None

        # Generate Plot from CSV
        if is_csv_mode:
            plot_path, data_path, verify_path, rulebased_status = (
                thirdparty.process.csv_to_plot.csv_to_plot.process_file(
                    file_path=str(data),
                    test_name=plot_keyword,
                    output_folder=str(output_plot_folder),
                )
            )
            logger.debug(f"[CSV Mode] Path: {plot_path}, Status: {rulebased_status}")

        # Collect read/write data
        rw_data_wrap = read_data if plot_keyword == "Linear Read" else write_data
        rw_data_wrap.append(
            ivit.InferInput(
                data_path=data_path,
                plot_path=plot_path,
                verify_path=verify_path,
                domain=domain,
                rule_verify=rulebased_status,
            )
        )

    return read_data, write_data
