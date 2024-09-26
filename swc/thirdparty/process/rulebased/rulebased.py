import pandas as pd


def filter_dataframe(
    data: pd.DataFrame, test_name: str = "Linear Write"
) -> pd.DataFrame:
    """
    Filters the given DataFrame to retain only the specified test's results and extracts the
    numeric MB/s values from the 'Result' column.

    Args:
        data (pd.DataFrame): The input DataFrame containing at least the 'Test' and 'Result' columns.
        test_name (str): The name of the test to filter by. Defaults to "Linear Write".

    Returns:
        pd.DataFrame: Returns a DataFrame with filtered rows and a new column 'Result_MBps'
        containing the numeric MB/s values. Returns None if the specified test name is not found.
    """

    # Filter the DataFrame to only include the 'Test' and 'Result' columns, and drop any rows with missing values
    data_filtered = data[["Test", "Result"]].dropna()

    # Filter the DataFrame to retain only rows where 'Test' matches the specified test_name
    data_filtered = data_filtered[data_filtered["Test"] == test_name]

    # Extract the MB/s value from the 'Result' column, keeping only the numeric part
    data_filtered["Result_MBps"] = data_filtered["Result"].apply(
        lambda x: float(x.split()[0])
    )

    # Reset the index of the filtered DataFrame and return it
    data_filtered = data_filtered.reset_index(drop=True)

    return data_filtered


def process(data_filtered: pd.DataFrame, test_name: str) -> tuple:
    """
    Arguments:
        data_filtered (DataFrame): The filtered pd.dataframe
        test_name (str): testing columns

    Return:
        resuls (tuple): (max_val, min_val, avg_val, threshold, last_90_percent_min_val, last_90_percent_avg, status)

    計算特定測試名稱的最大值、最小值、平均值、後90%數據的平均值、以及判斷後90%數據的最小值是否小於設定的閥值

    參數:
    data_filtered (DataFrame): 篩選後的數據
    test_name (str): 測試名稱

    返回:
    tuple: 返回最大值、最小值、平均值、後90%數據平均值、判斷閥值、和 Pass/Fail 判斷結果的元組
           如果沒有數據，返回一組預設值而不是 None。

    """

    if filtered_data.empty:
        return 0, 0, 0, 0, 0, 0, "No Data"

    filtered_data = data_filtered[data_filtered["Test"] == test_name]

    # 使用 pandas 的內建函數計算最大值、最小值和平均值
    max_val = round(filtered_data["Result_MBps"].max(), 1)
    min_val = round(filtered_data["Result_MBps"].min(), 1)
    avg_val = round(filtered_data["Result_MBps"].mean(), 1)

    # 計算後90%的數據平均值
    num_last_90_percent = int(len(filtered_data) * 0.9)
    last_90_percent_results = filtered_data["Result_MBps"].iloc[-num_last_90_percent:]
    last_90_percent_avg = round(last_90_percent_results.mean(), 1)

    # 設定後90%數據中的第一筆在原始數據中的位置
    start_index_of_last_90_percent = len(filtered_data) - num_last_90_percent
    first_value_of_last_90_percent = filtered_data["Result_MBps"].iloc[
        start_index_of_last_90_percent
    ]
    # print(
    #     f"後90%的第一筆數據是在原始數據中的第 {start_index_of_last_90_percent + 1} 筆，數值為 {first_value_of_last_90_percent}。"
    # )

    # 設定閥值為後90%數據平均值的85%，這樣可以檢測數據是否出現異常波動
    threshold = round(last_90_percent_avg * 0.85, 1)

    # 判斷後90%數據中的最小值是否小於設定的閥值
    last_90_percent_min_val = round(last_90_percent_results.min(), 1)
    if last_90_percent_min_val < threshold:
        status = "Fail"
    else:
        status = "Pass"

    return (
        max_val,
        min_val,
        avg_val,
        threshold,
        last_90_percent_min_val,
        last_90_percent_avg,
        last_90_percent_min_val >= threshold,
    )


def test_process():
    file_path = r"C:\Users\max_chang\Desktop\s-wc-web\data\2CY12312250010038_R.csv"
    data = pd.read_csv(file_path)

    for test_name in ["Linear Write", "Linear Read"]:
        print("-" * 10)

        data_filtered = filter_dataframe(data, test_name)

        (
            max_val,
            min_val,
            avg_val,
            threshold,
            last_90_percent_min_val,
            last_90_percent_avg,
            status,
        ) = process(data_filtered)

        # 結果資訊
        result_info = (
            f"{test_name}: Max: {max_val}, Min: {min_val}, Avg: {avg_val}, "
            f"Last_90%_Min: {last_90_percent_min_val}, Last_90%_Avg: {last_90_percent_avg}, Status: {status}"
        )
        print(result_info)


if __name__ == "__main__":
    test_process()  # Linear Write: Max: 2098.3, Min: 229.8, Avg: 355.1, Last_90%_Min: 240.5, Last_90%_Avg: 270.5, Status: Pass
