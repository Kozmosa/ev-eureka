import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

DATASET_FILE = 'data.csv'

try:
    data = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"Error: The file {DATASET_FILE} was not found. Please ensure it is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading {DATASET_FILE}: {e}")
    exit()

data = pd.get_dummies(data, columns=['brand'], drop_first=True, dtype=int)

brand_columns = [col for col in data.columns if 'brand_' in col]

X_columns = [
    'average_speed', 'avg_daily_charges', 'fatigue_driving_ratio',
    'late_night_trip_ratio', 'avg_late_night_trip_mileage',
    'high_temp_driving_ratio', 'battery_type_lfp', 'initial_battery_soc',
    'avg_charge_duration', 'insurance_commercial_third_party',
    'insurance_compulsory_third_party'
] + brand_columns

try:
    X = data[X_columns].astype(float)
    y = data['average_loss'].astype(float)
except KeyError as e:
    print(f"Error: A required column is missing from the data: {e}")
    print(f"Available columns: {data.columns.tolist()}")
    exit()


if y.le(0).any():
    print("Warning: The dependent variable 'average_loss' contains non-positive values.")
    print(f"Number of non-positive values: {y.le(0).sum()}")
    print("These rows will cause issues with a Log link function. Consider cleaning or filtering the data.")
    print("For demonstration, replacing non-positive values with a small positive number (0.01).")
    y = np.maximum(y, 0.01)


X = sm.add_constant(X)

if X.isnull().sum().sum() > 0:
    print("Warning: Independent variables (X) contain NaN values. Attempting to fill with median.")
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    if X.isnull().sum().sum() > 0:
        print("Error: NaN values still present in X after attempting to fill. Please clean the data.")
        exit()

if y.isnull().sum() > 0:
    print("Error: Dependent variable (y) contains NaN values. Please clean the data.")
    exit()

try:
    glm_model = sm.GLM(y, X, family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5))
    glm_results = glm_model.fit()
    print(glm_results.summary())

except Exception as e:
    print(f"模型拟合过程中发生错误: {e}")
    print("请检查数据是否存在问题，例如：")
    print(f"- 因变量 y 是否包含非正值 (对于 Log link)。 当前y中小于等于0的值数量: {y.le(0).sum()}")
    print("- 自变量 X 是否存在完全共线性或包含非数值类型。")
    print(f"  X中NaN数量: {X.isnull().sum().sum()}")
    print(f"  X的数据类型:\n{X.dtypes.value_counts()}")
    print("- 样本量是否足够。")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")