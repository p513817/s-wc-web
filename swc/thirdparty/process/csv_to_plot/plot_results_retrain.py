import os

import matplotlib.pyplot as plt
import numpy as np


def plot_results_retrain(
    data_filtered, test_name, file_name, max_val, min_val, output_folder, line_width=2
):
    """
    繪製沒有軸和標籤的簡潔圖表，並將圖表保存為PNG文件

    參數:
    data_filtered (DataFrame): 篩選後的數據
    test_name (str): 測試名稱
    file_name (str): 圖片文件名（不帶擴展名）
    max_val (float): 最大值
    min_val (float): 最小值
    output_folder (str): 圖片保存的文件夾路徑
    line_width (int): 線條寬度
    """
    # 固定圖片大小為1600x900像素
    plt.figure(figsize=(16, 9), dpi=100)

    # 根據測試名稱篩選數據並提取結果值
    values = data_filtered[data_filtered["Test"] == test_name]["Result_MBps"].values

    # 使用NumPy生成均勻分佈的數列來表示x軸，範圍為0到100
    x_values = np.linspace(0, 100, len(values))

    # 繪製結果曲線
    plt.plot(x_values, values, color="white", linewidth=line_width)
    plt.ylim([min(values) * 0.8, max(values) * 1.1])
    plt.xlim([0, 100])

    # 去除所有軸和文字
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # 設置背景顏色為黑色
    plt.gca().set_facecolor("black")
    plt.gcf().patch.set_facecolor("black")

    # 圖片儲存的操作，增加異常處理
    output_path = os.path.join(output_folder, f"{file_name}_Retrain.png")
    try:
        plt.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0)
        # print(f"圖片儲存成功: {output_path}")
    except IOError:
        # print(f"圖片儲存失敗: {e}")
        pass

    finally:
        plt.close()

    return output_path
