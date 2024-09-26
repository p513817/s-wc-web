import glob
from pathlib import Path

import pandas as pd

from ..rulebased import rulebased
from . import plot_results, plot_results_retrain
import logging

logger = logging.getLogger(__name__)

def process_file(
    file_path, test_name, output_folder, draw_red_rect=False
) -> tuple[str]:
    """
    Process CSV file and generate plot

    Arguments:
        - file_path (str): CSV文件的路徑
        - test_name (str): 要繪製圖表的測試名稱（例如 'Linear Write' 或 'Linear Read'）
        - output_folder (str): 圖片保存的文件夾路徑
        - draw_red_rect (bool): 是否繪製紅色矩形框的開關（預設為 False）

    Return:
        Tuple: result_path, result_no_axes_path, status
    """
    # 讀取CSV文件並加載到DataFrame中
    data = pd.read_csv(file_path)

    # 篩選出需要的欄位 'Test' 和 'Result'，並移除缺失值
    data_filtered = data[["Test", "Result"]].dropna()

    # 將 'Result' 欄位中的 MB/s 值提取出來，僅保留數值部分
    data_filtered["Result_MBps"] = data_filtered["Result"].apply(
        lambda x: float(x.split()[0])
    )

    # 重設索引
    data_filtered = data_filtered.reset_index(drop=True)

    # 獲取原始文件名（不帶擴展名）
    original_file_name = Path(file_path).stem

    # 檢查資料中是否包含指定的測試項目
    if test_name not in data_filtered["Test"].values:
        return

    (
        max_val,
        min_val,
        avg_val,
        threshold,
        last_90_percent_min_val,
        last_90_percent_avg,
        status,
    ) = rulebased.process(data_filtered, test_name)

    # 結果資訊
    result_info = (
        f"{test_name}: Max: {max_val}, Min: {min_val}, Avg: {avg_val}, "
        f"Last_90%_Min: {last_90_percent_min_val}, Last_90%_Avg: {last_90_percent_avg}, Thres: {threshold}, Status: {status}"
    )
    logger.debug(result_info)

    # 準備資料
    kwargs = {
        "data_filtered":data_filtered,
        "test_name":test_name,
        "file_name":original_file_name,
        "max_val":max_val,
        "min_val":min_val,
        "output_folder":output_folder,
    }

    # 生成並儲存圖表
    result_path = plot_results.plot_results(
        avg_val = avg_val,
        last_90_percent_avg = last_90_percent_avg,
        last_90_percent_min_val = last_90_percent_min_val,
        threshold = threshold,
        draw_red_rect=False,
        **kwargs,
    )
    verify_path = plot_results.plot_results(
        avg_val = avg_val,
        last_90_percent_avg = last_90_percent_avg,
        last_90_percent_min_val = last_90_percent_min_val,
        threshold = threshold,
        draw_red_rect=True,
        **kwargs,
    )
    result_no_axes_path = plot_results_retrain.plot_results_retrain(
        **kwargs
    )

    return (result_path, result_no_axes_path, verify_path, status)


def process(folder_path: str):
    folder_path = Path(folder_path)

    # 設定是否繪製紅色矩形框的全域變數
    draw_red_rect = True

    # 設置圖表保存的路徑
    output_folder = folder_path / "output"
    if output_folder.exists:
        output_folder.unlink(missing_ok=True)
    output_folder.mkdir(parents=True)

    # 查找當前資料夾中包含 "_W" 的 .csv 文件
    write_files = glob.glob(str(folder_path / "*_W*.csv"))
    read_files = glob.glob(str(folder_path / "*_R*.csv"))

    # 處理找到的 Write 文件
    read_plot, read_plot_retrain = [], []
    for write_file_path in write_files:
        result_path, result_no_axes_path, verify_path = process_file(
            write_file_path, "Linear Write", output_folder, draw_red_rect=draw_red_rect
        )
        read_plot.append(result_path)
        read_plot_retrain.append(result_no_axes_path)

    # 處理找到的 Read 文件
    write_plot, write_plot_retrain = [], []
    for read_file_path in read_files:
        result_path, result_no_axes_path, verify_path = process_file(
            read_file_path, "Linear Read", output_folder, draw_red_rect=draw_red_rect
        )
        write_plot.append(result_path)
        write_plot_retrain.append(result_no_axes_path)

    return ((read_plot, read_plot_retrain), (write_plot, write_plot_retrain))


if __name__ == "__main__":
    process(r"C:\Users\max_chang\Desktop\s-wc-web\data")
