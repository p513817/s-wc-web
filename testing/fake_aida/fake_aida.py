"""
測試用的 AIDA
1. 複製 當前路徑下的 資料夾 到 data 資料夾下 模擬 AIDA 執行期間產生的檔案
"""

import argparse
import pathlib as pl
import shutil

if __name__ == "__main__":
    # 初始化 ArgumentParser
    parser = argparse.ArgumentParser(description="Select output directory.")

    # 添加參數
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory where files will be saved.",
    )

    # 解析參數
    args = parser.parse_args()

    src = pl.Path(__file__).parent / "aida64v730_2CY12312250010038"
    dst = pl.Path(args.output) / "aida64v730_2CY12312250010038"

    if dst.exists():
        shutil.rmtree(dst)

    import time

    time.sleep(2)
    shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    if dst.exists():
        print("Success")
