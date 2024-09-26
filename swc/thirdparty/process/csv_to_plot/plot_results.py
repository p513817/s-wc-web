import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def configure_axes(
    ax, x_values, y_min, y_max, line_width, bg_color="black", axis_color="white"
):
    """
    配置圖表軸、背景顏色和軸線顏色的輔助函數

    參數:
    ax (Axes): Matplotlib 的軸對象
    x_values (list): x軸的數值
    y_min (float): y軸的最小值
    y_max (float): y軸的最大值
    line_width (int): 線條寬度
    bg_color (str): 圖表背景顏色
    axis_color (str): 軸線顏色
    """
    # 設定y軸和x軸的範圍
    ax.set_ylim([y_min, y_max])
    ax.set_xlim([0, 100])

    # 設置x軸和y軸的刻度標籤顏色
    ax.tick_params(axis="x", colors=axis_color)
    ax.tick_params(axis="y", colors=axis_color)

    # 設置圖表背景顏色
    ax.set_facecolor(bg_color)

    # 設置軸線顏色
    for spine in ax.spines.values():
        spine.set_color(axis_color)
        spine.set_linewidth(line_width)

    # 添加虛線
    for i in range(0, 101, 10):
        ax.axvline(x=i, color=axis_color, linestyle="--", linewidth=0.5)
    y_ticks = np.linspace(y_min, y_max, num=11)
    for val in y_ticks:
        ax.axhline(y=val, color=axis_color, linestyle="--", linewidth=0.5)


def add_red_rectangle(ax, x_values, values, threshold, last_90_percent_avg):
    """
    添加紅色矩形框來標記異常數據的輔助函數

    參數:
    ax (Axes): Matplotlib 的軸對象
    x_values (list): x軸的數值
    values (list): 測試結果值
    threshold (float): 判斷閥值
    last_90_percent_avg (float): 後90%數據的平均值
    """
    for i in range(len(values)):
        if x_values[i] > 10 and values[i] < threshold:
            # 如果 values[i] - 150 小於 0，則設置為 values[i],其餘設置values[i] - 30
            adjusted_value = values[i] if values[i] - 150 < 0 else values[i] - 30
            # 計算矩形框的高度
            rect_height = last_90_percent_avg - adjusted_value
            rect = patches.Rectangle(
                (x_values[i] - 1, adjusted_value),
                2,
                rect_height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)


def plot_results(
    data_filtered,
    test_name,
    file_name,
    max_val,
    min_val,
    avg_val,
    last_90_percent_avg,
    last_90_percent_min_val,
    threshold,
    output_folder,
    line_width=2,
    draw_red_rect=False,
    bg_color="black",
    axis_color="white",
):
    """
    繪製帶有標籤和軸刻度的圖表，並將圖表保存為PNG文件

    參數:
    data_filtered (DataFrame): 篩選後的數據
    test_name (str): 測試名稱
    file_name (str): 圖片文件名（不帶擴展名）
    max_val (float): 最大值
    min_val (float): 最小值
    avg_val (float): 平均值
    last_90_percent_avg (float): 後90%平均值
    threshold (float): 判斷閥值
    output_folder (str): 圖片保存的文件夾路徑
    line_width (int): 線條寬度
    draw_red_rect (bool): 是否繪製紅色矩形框的開關
    bg_color (str): 圖表背景顏色
    axis_color (str): 軸線顏色
    """

    # 創建一個新的圖表，並固定圖片大小為1600x900像素
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(bg_color)  # 設置背景顏色

    # 從篩選後的數據中，根據測試名稱提取對應的結果值
    values = data_filtered[data_filtered["Test"] == test_name]["Result_MBps"].values

    # 使用 np.linspace 生成 x 軸值的列表，範圍為 0 到 100
    x_values = np.linspace(0, 100, len(values))

    # 繪製結果曲線
    ax.plot(x_values, values, label="Result", color=axis_color, linewidth=line_width)

    # 定義 Y 軸上下限的擴展比例
    y_min_multiplier = 0.8
    y_max_multiplier = 1.1
    y_min = min_val * y_min_multiplier
    y_max = max_val * y_max_multiplier

    # 配置圖表軸和背景
    configure_axes(ax, x_values, y_min, y_max, line_width, bg_color, axis_color)

    # 設定圖表的標題和軸標籤
    ax.set_title(f"{test_name} - Result", color=axis_color, pad=23, fontsize=16)
    ax.set_xlabel("Completion (%)", color=axis_color)
    ax.set_ylabel("MB/s", color=axis_color)

    # 在標題下方添加 Max, Min, Avg 的數值
    plt.text(
        0.5,
        0.895,
        f"Max: {max_val}, Min: {min_val}, Avg: {avg_val}, 90% Avg: {last_90_percent_avg}, 90% Min: {last_90_percent_min_val}, Thres: {threshold}",
        ha="center",
        va="center",
        transform=fig.transFigure,
        color=axis_color,
        fontsize=12,
    )

    # 添加圖例
    ax.legend(loc="best", frameon=False, fontsize=12, labelcolor=axis_color)

    # 添加紅色矩形框來標記異常數據（如果啟用）
    if draw_red_rect:
        add_red_rectangle(ax, x_values, values, threshold, last_90_percent_avg)

    # 儲存圖片
    if draw_red_rect:
        file_name = file_name + "_Verify" 
    output_path = os.path.join(output_folder, f"{file_name}.png")
    plt.savefig(output_path, format="png", dpi=100, bbox_inches=None, pad_inches=0)
    plt.close()

    return output_path
