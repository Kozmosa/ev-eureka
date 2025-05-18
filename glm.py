import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# global environs
DATASET_DIR = 'datasets'
DATASET_FILE = os.path.join(DATASET_DIR, 'ev.csv')

# 1. 创建模拟数据集
# --------------------
# 根据文章摘要，GLM模型中使用的变量包括：
# 品牌 (brand), 平均速度 (average_speed), 日均充电次数 (avg_daily_charges),
# 疲劳驾驶行程数占比 (fatigue_driving_ratio), 深夜行程数占比 (late_night_trip_ratio),
# 深夜单次行程平均行驶里程 (avg_late_night_trip_mileage), 高温行驶时长占比 (high_temp_driving_ratio),
# 电池类型_磷酸铁锂电池 (battery_type_lfp) (二元变量，1表示磷酸铁锂，0表示其他),
# 行车起始剩余电量 (initial_battery_soc), 单次充电平均时长 (avg_charge_duration),
# 险种_交三 (insurance_commercial_third_party) (二元变量), 险种_单交 (insurance_compulsory_third_party) (二元变量)
# 因变量: 车均损失 (average_loss)

np.random.seed(42) # 为了结果可复现
num_samples = 1000

data = pd.DataFrame({
    'brand': np.random.choice(['品牌A', '品牌B', '品牌C', '品牌D'], num_samples),
    'average_speed': np.random.uniform(20, 80, num_samples), # 平均速度 km/h
    'avg_daily_charges': np.random.poisson(1, num_samples), # 日均充电次数
    'fatigue_driving_ratio': np.random.uniform(0, 0.1, num_samples), # 疲劳驾驶行程数占比
    'late_night_trip_ratio': np.random.uniform(0, 0.2, num_samples), # 深夜行程数占比
    'avg_late_night_trip_mileage': np.random.uniform(0, 50, num_samples) * (np.random.rand(num_samples) < 0.2), # 深夜单次行程平均行驶里程, 假设部分为0
    'high_temp_driving_ratio': np.random.uniform(0, 0.3, num_samples), # 高温行驶时长占比
    'battery_type_lfp': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]), # 1 if LFP, 0 otherwise
    'initial_battery_soc': np.random.uniform(20, 100, num_samples), # 行车起始剩余电量 %
    'avg_charge_duration': np.random.uniform(3600, 28800, num_samples), # 单次充电平均时长 (秒)
    'insurance_commercial_third_party': np.random.choice([0, 1], num_samples, p=[0.4, 0.6]),
    'insurance_compulsory_third_party': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]) # 假设交三和单交不是互斥的，或者需要更复杂的编码
})


# 生成因变量 "车均损失" (average_loss)
# 这是一个简化的模拟，实际关系会更复杂
# 我们将基于一些特征创建一个基础损失，并加入随机性
log_average_loss = (
    1.5  # Intercept
    + 0.01 * data['average_speed']
    - 0.1 * data['avg_daily_charges']
    + 5 * data['fatigue_driving_ratio']
    + 3 * data['late_night_trip_ratio']
    + 0.005 * data['avg_late_night_trip_mileage']
    + 2 * data['high_temp_driving_ratio']
    - 0.2 * data['battery_type_lfp']
    - 0.005 * data['initial_battery_soc']
    + 0.000001 * data['avg_charge_duration'] # 确保系数与特征的尺度相匹配
    + 0.3 * data['insurance_commercial_third_party']
    + 0.1 * data['insurance_compulsory_third_party']
    + np.random.normal(0, 0.5, num_samples) # 随机噪声
)
# 确保损失为正，并模拟Tweedie分布（这里用指数变换后的对数正态近似）
data['average_loss'] = np.exp(log_average_loss)
# 对于Tweedie，损失可以为0，但这里我们确保其为正以简化模拟
# 并且避免过小的值，因为log link对0或负值无效
data['average_loss'] = np.maximum(data['average_loss'], 0.01) # 设置一个很小的正数作为最小损失值

# 保存样例数据集
data.to_csv(DATASET_FILE)

# 2. 数据预处理
# -----------------
# 对分类变量 'brand' 进行独热编码，并确保生成整数类型
data = pd.get_dummies(data, columns=['brand'], drop_first=True, dtype=int) # drop_first=True 避免多重共线性

# 准备 X (自变量) 和 y (因变量)
# 从文章摘要中选择GLM模型中提到的变量作为自变量
# 注意：原文图4.5中包含了“品牌”作为整体变量，这里我们用独热编码后的品牌
# “电池类型_磷酸铁锂电池” 和 “险种” 已经是0/1编码或将通过独热编码处理

# 确保列名与独热编码后的名称一致
brand_columns = [col for col in data.columns if 'brand_' in col]

X_columns = [
    'average_speed', 'avg_daily_charges', 'fatigue_driving_ratio',
    'late_night_trip_ratio', 'avg_late_night_trip_mileage',
    'high_temp_driving_ratio', 'battery_type_lfp', 'initial_battery_soc',
    'avg_charge_duration', 'insurance_commercial_third_party',
    'insurance_compulsory_third_party'
] + brand_columns

X = data[X_columns].astype(float) # 确保所有X列都是数值类型，通常是float
y = data['average_loss'].astype(float) # 确保y也是float

# 添加常数项 (截距) 到自变量中
X = sm.add_constant(X)

# 检查 X 和 y 的数据类型
# print("Data types of X before fitting the model:")
# print(X.dtypes)
# print("\nData types of y before fitting the model:")
# print(y.dtypes)
# print("\nChecking for NaNs in X:")
# print(X.isnull().sum().sum())
# print("\nChecking for NaNs in y:")
# print(y.isnull().sum())
# print("\nChecking for non-positive values in y (for Log link):")
# print((y <= 0).sum())


# 3. 构建和拟合 GLM 模型
# -------------------------
# 根据文章，因变量“车均损失”在预测纯风险保费时，更接近于 Tweedie 分布。
# 连接函数通常使用 log。
# Tweedie 分布的 variance power (p) 通常在 1 到 2 之间。
# p=1 对应 Poisson, p=2 对应 Gamma。对于保险纯损失，p 常取 1.5 左右。
# statsmodels 中的 Tweedie(link=sm.families.links.Log(), var_power=<p>)
# 如果文章未指定 var_power，我们将使用一个常见值，例如 1.5。

try:
    # 定义模型：Tweedie 分布族 和 Log 连接函数 (注意大写 L)
    # var_power 是 Tweedie 分布的一个关键参数，需要根据数据特性或领域知识设定
    # 文章中没有明确给出 var_power 的值，这里我们假设一个常用的值 1.5
    glm_model = sm.GLM(y, X, family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5))
    
    # 拟合模型
    glm_results = glm_model.fit()
    
    # 4. 打印模型摘要
    # -----------------
    print(glm_results.summary())

except Exception as e:
    print(f"模型拟合过程中发生错误: {e}")
    print("请检查数据是否存在问题，例如：")
    print("- 因变量 y 是否包含非正值 (对于 Log link)。 当前y中非正值数量:", (y <= 0).sum())
    print("- 自变量 X 是否存在完全共线性或包含非数值类型。")
    print("  X中NaN数量:", X.isnull().sum().sum())
    print("  X的数据类型:", X.dtypes.value_counts())
    print("- 样本量是否足够。")

# 进一步分析 (可选)
# --------------------
# 例如，可以查看系数、置信区间等
# print("\n模型系数:")
# print(glm_results.params)
#
# print("\n系数置信区间:")
# print(glm_results.conf_int())

# 注意:
# 1. 上述代码中的数据生成是高度简化的，仅用于演示目的。
#    真实数据将具有更复杂的分布和变量间关系。
# 2. 变量的选择和转换（例如，连续变量的分箱、交互项的创建）在实际建模中非常重要，
#    应基于文章中更详细的数据预处理步骤（如果提供）或领域知识。
#    文章摘要中提到“自变量是否标准化：是”，在实际应用中，如果使用正则化或某些优化算法，标准化可能是有益的。
#    对于标准GLM，系数的解释会因子变量的尺度而变化，但模型拟合本身不一定需要标准化。
# 3. `var_power` for Tweedie: 这个参数对模型结果有显著影响。在实际应用中，
#    通常需要通过交叉验证或最大似然估计等方法来选择最佳的 `var_power`。
#    文章中没有提供这个参数，因此我们选择了一个常用的值。
# 4. 险种编码：如果“险种_交三”和“险种_单交”是某个基础“险种”变量的多个级别，
#    那么独热编码可能是合适的。如果它们代表不同的附加险，当前的0/1编码可能是正确的。
#    根据摘要，它们似乎是独立的二元变量。
# 5. 确保 y (因变量) 对于 Log link 来说是严格为正的。我在生成y后添加了 np.maximum(data['average_loss'], 0.01) 来处理这个问题。
# 6. 确保 X 中的所有列都是数值类型，通过 .astype(float) 强制转换。
