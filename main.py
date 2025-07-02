import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import glob
import argparse


# 1. 数据清洗

# 填补缺失数据（线性插值）
def fill_missing_data(df):
    return df.interpolate(method='linear', axis=0)


# 识别并修复错误数据（异常值处理）
def handle_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < 3).all(axis=1)]  # 过滤掉超过3个标准差的异常值
    return df


# 噪声去除（低通滤波器）
def lowpass_filter(data, cutoff=0.1, fs=1.0, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# 统一格式转换（将不同的测试数据格式转换为统一的CSV格式）
def convert_to_csv(file_path, output_dir):
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.dat') or file_path.endswith('.txt'):
        df = pd.read_csv(file_path, delimiter='\t')  # 假设分隔符是tab
    else:
        df = pd.read_csv(file_path)

    output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.dat', '.csv').replace('.xlsx', '.csv'))
    df.to_csv(output_path, index=False)
    print(f'Converted {file_path} to {output_path}')


# 2. 设备误差校正

# 假设我们通过线性回归校正设备误差
def calibrate_device(df, columns_to_calibrate):
    for column in columns_to_calibrate:
        # 假设误差源为温度，我们通过温度（假设存在一个列'Temperature'）来校正电压
        X = df[['Temperature']].dropna()  # 温度列
        y = df[column].dropna()  # 待校正列
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        df[column] = df[column] - (y - y_pred)  # 校正误差
    return df


# 3. 容量计算

# 计算电池的容量（mAh），通过积分充放电曲线
def calculate_capacity(df, time_column='Time', current_column='Current'):
    time_interval = np.diff(df[time_column]) / 3600  # 转换为小时
    current = df[current_column].values[1:]
    capacity = np.sum(current * time_interval)  # 积分，得到电池的容量
    return capacity


# 4. 循环寿命计算

# 基于电压变化计算电池的循环寿命
def calculate_cycle_lifetime(df, voltage_column='Voltage', cycle_threshold=3.0):
    cycles = (df[voltage_column].shift(1) > cycle_threshold) & (df[voltage_column] <= cycle_threshold)
    cycle_lifetime = cycles.sum()  # 统计跌破阈值的循环次数
    return cycle_lifetime


# 5. 特征工程

# 计算电流变化率（电流与时间的变化率）
def calculate_current_rate(df, current_column='Current', time_column='Time'):
    df['Current_Rate'] = np.diff(df[current_column]) / np.diff(df[time_column])
    df['Current_Rate'] = np.insert(df['Current_Rate'], 0, 0)  # 插入第一个值作为0（没有变化率）
    return df


# 6. 机器学习模型集成

# 训练一个简单的随机森林回归模型，预测电池容量和寿命
def train_model(df, target_column, feature_columns):
    # 特征和目标变量
    X = df[feature_columns].dropna()
    y = df[target_column].dropna()

    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 训练随机森林回归模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 模型评估
    test_score = model.score(X_test, y_test)
    print(f'Model test score: {test_score:.2f}')

    return model, scaler


# 7. 数据处理

# 处理单个文件
def process_file(file_path, output_dir, columns_to_calibrate, model=None, scaler=None, feature_columns=None):
    df = pd.read_csv(file_path)

    # 数据清洗
    df = fill_missing_data(df)
    df = handle_outliers(df)

    # 数据校正
    df = calibrate_device(df, columns_to_calibrate)

    # 数据平滑
    df = df.apply(
        lambda x: lowpass_filter(x, cutoff=0.1) if x.name in df.select_dtypes(include=[np.number]).columns else x)

    # 计算容量
    capacity = calculate_capacity(df)
    print(f'Calculated capacity: {capacity:.2f} mAh')

    # 计算循环寿命
    cycle_lifetime = calculate_cycle_lifetime(df)
    print(f'Calculated cycle lifetime: {cycle_lifetime} cycles')

    # 特征工程：计算电流变化率
    df = calculate_current_rate(df)

    # 预测容量或寿命（如果模型已训练）
    if model and scaler and feature_columns:
        X = df[feature_columns].dropna()
        X_scaled = scaler.transform(X)
        predicted_value = model.predict(X_scaled)
        print(f'Predicted {target_column}: {predicted_value[0]:.2f}')

    # 输出清洗和处理后的数据
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    df.to_csv(output_file, index=False)
    print(f'Processed file saved to {output_file}')


# 批量处理函数
def process_batch(input_dir, output_dir, columns_to_calibrate, model=None, scaler=None, feature_columns=None):
    files = glob.glob(os.path.join(input_dir, '*'))
    for file in files:
        process_file(file, output_dir, columns_to_calibrate, model, scaler, feature_columns)


# 8. 主程序接口（命令行）

def main():
    parser = argparse.ArgumentParser(description='Battery Data Processing Script')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing the raw data files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory to save processed files')
    parser.add_argument('--columns-to-calibrate', type=str, nargs='+', required=True,
                        help='Columns to apply device calibration')
    parser.add_argument('--train-model', action='store_true', help='Whether to train a machine learning model')

    args = parser.parse_args()

    # 训练机器学习模型
    model = None
    scaler = None
    feature_columns = None
    if args.train_model:
        # 假设我们用Voltage, Current作为特征列，Capacity作为目标列来训练模型
        feature_columns = ['Voltage', 'Current', 'Temperature', 'Current_Rate']
        target_column = 'Capacity'  # 以电池容量为目标变量
        df = pd.read_csv(os.path.join(args.input_dir, os.listdir(args.input_dir)[0]))  # 假设至少有一个文件
        model, scaler = train_model(df, target_column, feature_columns)

    # 批量处理数据
    process_batch(args.input_dir, args.output_dir, args.columns_to_calibrate, model, scaler, feature_columns)


if __name__ == '__main__':
    main()
