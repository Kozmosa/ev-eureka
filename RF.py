import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer # Import SimpleImputer
import shap # 需要安装 shap 库: pip install shap
import joblib
import json

# 变量参数值
DATASET_DIR = 'datasets'
# DATASET_FILE = os.path.join(DATASET_DIR, 'car.csv')
DATASET_FILE = os.path.join(DATASET_DIR, 'ev.csv')
# TARGET_COLUMN = "CLAIM_PAID"
TARGET_COLUMN = "average_loss"
MODELS_DIR = 'models' # 文件夹名称
MODEL_FILE = os.path.join(MODELS_DIR, 'rf_model.joblib')
SCALER_FILE = os.path.join(MODELS_DIR, 'rf_scaler.joblib')
IMPUTER_FILENAME = os.path.join(MODELS_DIR, 'rf_imputer.joblib') # For the imputer
FEATURE_COLUMNS_FILENAME = os.path.join(MODELS_DIR, 'rf_feature_columns.json') # For feature names


# 模型保存和模型加载函数
def save_model_components(model, scaler, imputer, feature_columns_list,
                          model_fn, scaler_fn, imputer_fn, columns_fn):
    """Saves the model, scaler, imputer, and feature column list."""
    os.makedirs(os.path.dirname(model_fn), exist_ok=True)
    try:
        joblib.dump(model, model_fn)
        joblib.dump(scaler, scaler_fn)
        joblib.dump(imputer, imputer_fn)
        with open(columns_fn, 'w') as f:
            json.dump(feature_columns_list, f)
        print(f"Model, scaler, imputer, and feature columns successfully saved to '{MODELS_DIR}'.")
        return True
    except Exception as e:
        print(f"Error saving components: {e}")
        return False

def load_model_components(model_fn, scaler_fn, imputer_fn, columns_fn):
    """Loads the model, scaler, imputer, and feature column list."""
    try:
        model = joblib.load(model_fn)
        scaler = joblib.load(scaler_fn)
        imputer = joblib.load(imputer_fn)
        with open(columns_fn, 'r') as f:
            feature_columns_list = json.load(f)
        print(f"Model, scaler, imputer, and feature columns successfully loaded from '{MODELS_DIR}'.")
        return model, scaler, imputer, feature_columns_list
    except Exception as e:
        print(f"Error loading components: {e}")
        return None, None, None, None

def save_model_and_scaler(model, scaler, model_filename: str, scaler_filename: str):
    """
    使用 joblib 保存 scikit-learn 模型和预处理器 (scaler)。

    参数:
    model: 已经训练好的 scikit-learn 模型对象。
    scaler: 已经 fit 好的 scikit-learn 预处理器对象 (例如 StandardScaler)。
    model_filename (str): 模型保存的文件路径和名称 (例如 'model.joblib')。
    scaler_filename (str): 预处理器保存的文件路径和名称 (例如 'scaler.joblib')。
    """
    try:
        joblib.dump(model, model_filename)
        print(f"模型已成功保存到: {model_filename}")
    except Exception as e:
        print(f"保存模型失败: {e}")
        return False

    try:
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler 已成功保存到: {scaler_filename}")
    except Exception as e:
        print(f"保存 Scaler 失败: {e}")
        # 如果模型保存成功但 scaler 失败，可能需要考虑回滚或特殊处理
        return False
    return True

def load_model_and_scaler(model_filename: str, scaler_filename: str):
    """
    使用 joblib 加载 scikit-learn 模型和预处理器 (scaler)。

    参数:
    model_filename (str): 模型保存的文件路径和名称。
    scaler_filename (str): 预处理器保存的文件路径和名称。

    返回:
    tuple: (loaded_model, loaded_scaler) 如果成功，否则 (None, None)。
    """
    loaded_model = None
    loaded_scaler = None
    try:
        loaded_model = joblib.load(model_filename)
        print(f"模型已从 {model_filename} 成功加载。")
    except FileNotFoundError:
        print(f"错误: 模型文件 {model_filename} 未找到。")
    except Exception as e:
        print(f"加载模型失败: {e}")

    try:
        loaded_scaler = joblib.load(scaler_filename)
        print(f"Scaler 已从 {scaler_filename} 成功加载。")
    except FileNotFoundError:
        print(f"错误: Scaler 文件 {scaler_filename} 未找到。")
    except Exception as e:
        print(f"加载 Scaler 失败: {e}")

    return loaded_model, loaded_scaler

def sim_train():
    # 1. 生成模拟数据
    # 根据描述：数据量：93332 条保单记录, 自变量数量：37
    n_samples = 93332
    n_features = 37

    # 生成随机自变量数据
    X = pd.DataFrame(np.random.rand(n_samples, n_features),
                    columns=[f'feature_{i+1}' for i in range(n_features)])

    # 生成随机因变量数据 (车均损失) - 这里只是模拟，实际关系会更复杂
    # 为了让 min_samples_leaf=800 有意义，我们让 y 与部分特征有一定关系，并加入噪声
    np.random.seed(42) # for reproducibility
    y = (X['feature_1'] * 10 +
        X['feature_2'] * 5 -
        X['feature_3'] * 7 +
        np.random.randn(n_samples) * 20 + # 较大的噪声
        50) # 基准值
    y = pd.Series(y, name='车均损失')

    print(f"Generated data: X shape {X.shape}, y shape {y.shape}")
    print("\nSample of X:")
    print(X.head())
    print("\nSample of y:")
    print(y.head())

    # 2. 数据预处理
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 自变量标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将标准化后的数据转回 DataFrame (可选，主要为了 SHAP 显示特征名)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print(f"\nTrain data shape: X_train_scaled {X_train_scaled_df.shape}, y_train {y_train.shape}")
    print(f"Test data shape: X_test_scaled {X_test_scaled_df.shape}, y_test {y_test.shape}")

    # 3. 随机森林模型参数 (根据描述)
    rf_params = {
        'n_estimators': 100,         # 树的棵数
        'max_features': 0.6,         # 每次对自变量集合抽样的最大比例
        'max_samples': 0.7,          # 每次随机有放回抽样的最大比例 (bootstrap=True is default)
        'max_depth': 9,              # 树的最大深度
        'min_samples_leaf': 800,     # 叶结点所包含的最小样本数
        'random_state': 42,          # 为了结果可复现
        'oob_score': True,           # 使用袋外样本来估计泛化精度 (可选，但对Bagging方法有益)
        'n_jobs': -1                 # 使用所有可用的处理器核心 (加速训练)
    }

    # 初始化随机森林回归模型
    rf_model = RandomForestRegressor(**rf_params)

    print(f"\nInitializing RandomForestRegressor with parameters: {rf_params}")

    # 训练模型
    print("\nTraining the RandomForestRegressor model...")
    rf_model.fit(X_train_scaled_df, y_train)
    print("Model training complete.")

    # 保存模型
    save_success = save_model_and_scaler(rf_model, scaler, MODEL_FILE, SCALER_FILE)
    if not save_success:
        print("保存过程中发生错误，后续步骤可能失败。\n")
    else:
        print("模型和Scaler已保存。\n")

    # (可选) 评估模型
    y_pred_train = rf_model.predict(X_train_scaled_df)
    y_pred_test = rf_model.predict(X_test_scaled_df)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    oob_score = rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') else "N/A (oob_score=False or not available)"


    print(f"\nTraining MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"OOB Score: {oob_score if isinstance(oob_score, str) else oob_score:.4f}")


    # 4. 特征重要性
    print("\nExtracting feature importances...")
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10))

    # 5. SHAP (SHapley Additive exPlanations)
    print("\nCalculating SHAP values for model explanation...")
    # 对于树模型，TreeExplainer 更高效且精确
    explainer = shap.TreeExplainer(rf_model)

    # 计算 SHAP 值。为节省时间，通常在测试集的一个小子集上计算
    # 但这里我们用整个测试集，因为数据量不算特别巨大
    # 如果 X_test_scaled_df 非常大，可以考虑 shap_values = explainer.shap_values(X_test_scaled_df.sample(1000))
    shap_values = explainer.shap_values(X_test_scaled_df)

    print("SHAP values calculated.")

    # 可视化SHAP值
    # (a) Summary Plot: 显示每个特征对模型输出的总体影响
    print("\nGenerating SHAP summary plot (this may open a new window or display inline depending on your environment)...")
    shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=True) # bar plot for overall importance
    shap.summary_plot(shap_values, X_test_scaled_df, show=True) # dot plot for direction and magnitude

    # (b) Dependence Plot: 显示单个特征对模型输出的影响，以及它与另一个特征的交互作用
    # 例如，查看第一个特征 'feature_1' 的依赖图
    # 如果特征数量较多，选择最重要的几个特征进行分析
    if 'feature_1' in X_test_scaled_df.columns:
        print("\nGenerating SHAP dependence plot for 'feature_1' (example)...")
        shap.dependence_plot("feature_1", shap_values, X_test_scaled_df, interaction_index=None, show=True)

        # 如果想看 'feature_1' 和 'feature_2' 的交互
        if 'feature_2' in X_test_scaled_df.columns:
            shap.dependence_plot("feature_1", shap_values, X_test_scaled_df, interaction_index="feature_2", show=True)
    else:
        print("\nSkipping dependence plot as 'feature_1' not found (e.g., if only 1 feature).")


    print("\n--- Script Finished ---")

def train(csv_filepath: str, target_column_name: str):
    """
    从 CSV 文件加载数据，训练随机森林模型，并保存模型和 scaler。

    参数:
    csv_filepath (str): CSV 数据文件的路径。
    target_column_name (str): CSV 文件中因变量（目标）列的名称。
    """
    # 1. 加载数据
    print(f"--- 步骤 1: 从 {csv_filepath} 加载数据 ---")
    try:
        data_df = pd.read_csv(csv_filepath)
        print(f"数据加载成功。数据形状: {data_df.shape}")
        print("\n数据预览 (前5行):")
        print(data_df.head())
    except FileNotFoundError:
        print(f"错误: 数据文件 {csv_filepath} 未找到。请检查文件路径和名称。")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # 检查目标列是否存在
    if target_column_name not in data_df.columns:
        print(f"错误: 目标列 '{target_column_name}' 在CSV文件中未找到。")
        print(f"可用列为: {data_df.columns.tolist()}")
        return

    print(f"\n目标变量 (因变量): '{target_column_name}'")

    # 2. 定义特征 (X) 和目标 (y)
    # 假设所有非目标列都是特征，并且都是数值型或已经预处理好
    # 如果有非数值特征或需要排除的ID列，需要在这里处理
    try:
        y = data_df[target_column_name]
        X = data_df.drop(columns=[target_column_name])
        print(f"特征 (X) 的形状: {X.shape}")
        print(f"目标 (y) 的形状: {y.shape}")
        print("\n特征列 (前5列):")
        print(X.columns[:5].tolist())
    except Exception as e:
        print(f"定义特征和目标时发生错误: {e}")
        return

    # 确保所有特征都是数值类型 (一个简单的检查)
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns
    if len(non_numeric_cols) > 0:
        print(f"\n警告: 以下特征列不是数值类型，可能导致问题: {non_numeric_cols.tolist()}")
        print("请在传入模型前进行独热编码或标签编码等预处理。")
        # 或者在此处添加更复杂的预处理逻辑
        # 为简单起见，我们在这里尝试将它们转换为数值，如果失败则报错
        try:
            X = X.apply(pd.to_numeric, errors='coerce')
            if X.isnull().sum().sum() > 0: # 检查是否有因转换失败产生的 NaN
                print("警告: 部分非数值列在转换为数值时产生了NaN值，请检查数据。")
                # 可以选择填充NaN或在此处停止
                # X = X.fillna(X.mean()) # 例如，用均值填充
        except Exception as e:
            print(f"转换非数值列为数值时出错: {e}")
            # return # 如果希望严格处理，则取消注释

    # 3. 数据预处理
    print("\n--- 步骤 2: 数据预处理 ---")
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 自变量标准化
    scaler = StandardScaler()
    # fit scaler 应该只在训练数据上进行，并且只对数值型特征进行
    # 假设此时 X_train 已经是全数值的 DataFrame
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    # 将标准化后的数据转回 DataFrame (保留列名，这对于后续SHAP等很重要)
    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_array, columns=X_test.columns, index=X_test.index)

    print(f"\n训练集形状: X_train_scaled {X_train_scaled_df.shape}, y_train {y_train.shape}")
    print(f"测试集形状: X_test_scaled {X_test_scaled_df.shape}, y_test {y_test.shape}")

    # 4. 随机森林模型参数 (根据描述)
    print("\n--- 步骤 3: 模型训练 ---")
    rf_params = {
        'n_estimators': 100,
        'max_features': 0.6,
        'max_samples': 0.7,
        'max_depth': 9,
        'min_samples_leaf': 800, # 请确保这个值相对于您的数据量是合理的
        'random_state': 42,
        'oob_score': True,
        'n_jobs': -1
    }
    # 检查 min_samples_leaf 是否过大
    if rf_params['min_samples_leaf'] > len(X_train_scaled_df) * (1-0.7) / 2 and rf_params['max_samples'] is not None : # 粗略估计叶子节点可能的最小样本量
        print(f"警告: min_samples_leaf ({rf_params['min_samples_leaf']}) 可能相对于训练样本量和max_samples设置过大，可能导致模型无法有效构建。")
    if rf_params['min_samples_leaf'] * 2 > len(X_train_scaled_df): # 如果叶节点数量大于总样本一半
         print(f"警告: min_samples_leaf ({rf_params['min_samples_leaf']}) 可能相对于训练样本量 ({len(X_train_scaled_df)}) 过大。")


    rf_model = RandomForestRegressor(**rf_params)
    print(f"\n初始化 RandomForestRegressor，参数: {rf_params}")

    # 训练模型
    print("\n开始训练 RandomForestRegressor 模型...")
    try:
        rf_model.fit(X_train_scaled_df, y_train)
        print("模型训练完成。")
    except Exception as e:
        print(f"模型训练过程中发生错误: {e}")
        return

    # 5. 保存模型和 Scaler
    print("\n--- 步骤 4: 保存模型和 Scaler ---")
    save_success = save_model_and_scaler(rf_model, scaler, MODEL_FILE, SCALER_FILE)
    if not save_success:
        print("保存模型/Scaler过程中发生错误。\n")
    else:
        print("模型和Scaler已成功保存。\n")

    # 6. (可选) 评估模型
    print("\n--- 步骤 5: 模型评估 ---")
    y_pred_train = rf_model.predict(X_train_scaled_df)
    y_pred_test = rf_model.predict(X_test_scaled_df)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    oob_score = rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') and rf_model.oob_score_ else "N/A"

    print(f"\n训练集 MSE: {train_mse:.4f}")
    print(f"测试集 MSE: {test_mse:.4f}")
    if isinstance(oob_score, str):
        print(f"OOB Score: {oob_score}")
    else:
        print(f"OOB Score: {oob_score:.4f}")


    # 7. (可选) 特征重要性
    print("\n--- 步骤 6: 特征重要性 ---")
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns, # 使用原始X的列名
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 特征重要性:")
    print(importance_df.head(10))

    # 8. (可选) SHAP 解释
    print("\n--- 步骤 7: SHAP 模型解释 ---")
    print("\n计算 SHAP 值...")
    try:
        explainer = shap.TreeExplainer(rf_model)
        # 为节省时间，可以在测试集的一个小子集上计算SHAP值
        # 例如: shap_sample_data = X_test_scaled_df.sample(min(1000, len(X_test_scaled_df)), random_state=42)
        shap_sample_data = X_test_scaled_df
        shap_values = explainer.shap_values(shap_sample_data)
        print("SHAP 值计算完成。")

        print("\n生成 SHAP summary plot (条形图)...")
        shap.summary_plot(shap_values, shap_sample_data, plot_type="bar", show=False) # show=False 避免打断流程
        # plt.savefig('shap_summary_bar.png') # 如果需要保存图像
        # plt.close()

        print("\n生成 SHAP summary plot (散点图)...")
        shap.summary_plot(shap_values, shap_sample_data, show=False)
        # plt.savefig('shap_summary_dot.png')
        # plt.close()
        print("SHAP 图已生成（如果 matplotlib 配置为内联显示，则可能已显示）。若要保存，请取消注释 plt.savefig 行。")

        # (可选) Dependence Plot
        if len(X.columns) > 0:
            top_feature = importance_df['Feature'].iloc[0]
            print(f"\n生成 SHAP dependence plot for '{top_feature}'...")
            shap.dependence_plot(top_feature, shap_values, shap_sample_data, interaction_index=None, show=False)
            # plt.savefig(f'shap_dependence_{top_feature}.png')
            # plt.close()
    except Exception as e:
        print(f"计算或绘制 SHAP 图时发生错误: {e}")


    print("\n--- 训练函数执行完毕 ---")

def evaluate_model(model: RandomForestRegressor,
                   X_test_scaled_df: pd.DataFrame, y_test: pd.Series,
                   feature_column_names: list,
                   X_train_scaled_df: pd.DataFrame = None, y_train: pd.Series = None): # 训练数据设为可选
    """
    评估模型，计算指标，显示特征重要性并进行SHAP分析。
    如果提供了训练数据，也会计算训练集指标。
    """
    print("\n--- 开始模型评估 ---")

    # 1. 计算指标
    print("\n计算评估指标...")
    if X_train_scaled_df is not None and y_train is not None:
        y_pred_train = model.predict(X_train_scaled_df)
        train_mse = mean_squared_error(y_train, y_pred_train)
        print(f"训练集 MSE: {train_mse:.4f}")
    else:
        print("未提供训练数据，跳过训练集 MSE 计算。")

    y_pred_test = model.predict(X_test_scaled_df)
    test_mse = mean_squared_error(y_test, y_pred_test)
    print(f"测试集 MSE: {test_mse:.4f}")

    oob_score_value = "N/A"
    if hasattr(model, 'oob_score') and model.oob_score: # 检查 oob_score 参数是否为 True
        if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
            oob_score_value = model.oob_score_
            print(f"OOB Score: {oob_score_value:.4f}")
        else:
            print("OOB Score: N/A (oob_score_ 属性不可用，即使 oob_score=True)")
    else:
        print("OOB Score: N/A (模型训练时 oob_score 未设置为 True 或不可用)")


    # 2. 特征重要性
    print("\n提取特征重要性...")
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_column_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        print("\nTop 10 特征重要性:")
        print(importance_df.head(10))
    else:
        print("模型没有 feature_importances_ 属性。")

    # 3. SHAP 模型解释
    print("\n进行 SHAP 模型解释...")
    if not X_test_scaled_df.empty:
        try:
            explainer = shap.TreeExplainer(model)
            sample_size = min(1000, len(X_test_scaled_df))
            shap_sample_data = X_test_scaled_df.sample(sample_size, random_state=42) if sample_size > 0 and len(X_test_scaled_df) >= sample_size else (X_test_scaled_df if len(X_test_scaled_df) > 0 else pd.DataFrame())

            if not shap_sample_data.empty:
                print(f"为SHAP分析抽样 {len(shap_sample_data)} 条数据...")
                shap_values = explainer.shap_values(shap_sample_data)
                print("SHAP 值计算完成。")
                shap.summary_plot(shap_values, shap_sample_data, plot_type="bar", show=True)
                shap.summary_plot(shap_values, shap_sample_data, show=True)
                print("SHAP 图已生成（若需显示，请取消注释 evaluate_model 中的相关行并设置 show=True）。")
            else:
                print("SHAP分析已跳过，因为测试样本为空或采样后为空。")
        except Exception as e:
            print(f"计算或绘制 SHAP 图时发生错误: {e}")
    else:
        print("SHAP分析已跳过，因为X_test_scaled_df为空。")
    print("\n--- 模型评估函数执行完毕 ---")


# --- 新函数：加载模型并进行评估 ---
def load_and_evaluate(csv_filepath: str, target_column_name: str,
                      model_fn: str, scaler_fn: str, imputer_fn: str, columns_fn: str):
    """
    加载已保存的模型和预处理组件，准备测试数据，并对模型进行评估。
    """
    print("--- 开始加载模型和组件进行评估 ---")
    loaded_model, loaded_scaler, loaded_imputer, loaded_feature_columns = load_model_components(
        model_fn, scaler_fn, imputer_fn, columns_fn
    )

    if not all([loaded_model, loaded_scaler, loaded_imputer, loaded_feature_columns]):
        print("一个或多个模型组件加载失败。评估中止。")
        return

    print("\n--- 准备测试数据进行评估 ---")
    try:
        full_data_df = pd.read_csv(csv_filepath)
        print(f"原始数据 '{csv_filepath}' 加载成功。形状: {full_data_df.shape}")
    except FileNotFoundError:
        print(f"错误: 数据文件 {csv_filepath} 未找到。")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    if target_column_name not in full_data_df.columns:
        print(f"错误: 目标列 '{target_column_name}' 在CSV文件中未找到。")
        return

    full_data_df[target_column_name] = full_data_df[target_column_name].fillna(0)
    y_full = full_data_df[target_column_name]
    X_full = full_data_df.drop(columns=[target_column_name])

    # --- 应用与训练时相同的特征工程 ---
    # 1. 日期特征
    date_cols = ['INSR_BEGIN', 'INSR_END'] # 这些是原始列名
    for col in date_cols:
        if col in X_full.columns:
            X_full[col] = pd.to_datetime(X_full[col], errors='coerce')
            X_full[f'{col}_YEAR'] = X_full[col].dt.year
            X_full[f'{col}_MONTH'] = X_full[col].dt.month
            X_full[f'{col}_DAY'] = X_full[col].dt.day
    if 'INSR_BEGIN' in X_full.columns and 'INSR_END' in X_full.columns and \
       pd.api.types.is_datetime64_any_dtype(X_full['INSR_BEGIN']) and \
       pd.api.types.is_datetime64_any_dtype(X_full['INSR_END']):
        X_full['INSURANCE_DURATION_DAYS'] = (X_full['INSR_END'] - X_full['INSR_BEGIN']).dt.days
    X_full = X_full.drop(columns=date_cols, errors='ignore')

    # 2. SEX 列处理
    if 'SEX' in X_full.columns and X_full['SEX'].dtype == 'object':
        X_full['SEX'] = pd.to_numeric(X_full['SEX'], errors='coerce')

    # 3. 独热编码 (使用与训练时相同的思路确定列)
    # 注意：这里我们不知道训练时具体哪些高基数特征被丢弃了，
    # 但 get_dummies 后使用 loaded_feature_columns.reindex 会处理这个问题。
    # 我们需要的是那些 *原本* 是分类的列名。
    # 为了简化，我们假设训练时用于OHE的列是基于其dtype或已知列表。
    # 在实际部署中，最好保存训练时用于OHE的原始分类列名列表。
    potential_categorical_to_encode = []
    known_categoricals = ['EFFECTIVE_YR', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
    if 'SEX' in X_full.columns and X_full['SEX'].dtype == 'object':
         known_categoricals.append('SEX')

    for col in X_full.columns: # 检查 X_full 中当前的列
        if X_full[col].dtype == 'object' or col in known_categoricals:
            if col not in potential_categorical_to_encode:
                potential_categorical_to_encode.append(col)
    
    # 确保这些列真的存在于X_full中
    actual_categorical_to_encode = [col for col in potential_categorical_to_encode if col in X_full.columns]

    if actual_categorical_to_encode:
        print(f"对测试数据进行独热编码的类别特征: {actual_categorical_to_encode}")
        X_full_fe = pd.get_dummies(X_full, columns=actual_categorical_to_encode, dummy_na=False, dtype=np.uint8)
    else:
        X_full_fe = X_full.copy()
        print("测试数据中没有类别特征进行独热编码。")
    
    # 确保所有原始数值列如果不是float，也转换一下，以防万一
    for col in X_full_fe.columns:
        if col not in loaded_feature_columns and X_full_fe[col].dtype not in ['float64', 'float32', 'uint8', 'int8', 'int16', 'int32', 'int64']: # 假设dummy是uint8
            is_bool = pd.api.types.is_bool_dtype(X_full_fe[col])
            if not is_bool: # Don't convert bools to float unnecessarily
                try:
                    X_full_fe[col] = pd.to_numeric(X_full_fe[col], errors='coerce')
                except: pass # Ignore if it fails, imputer might catch it if it results in NaN


    # 4. 列对齐 (非常重要!)
    # 使用加载的特征列名来确保测试集与训练集有相同的列结构
    # 新数据中没有的列会以0填充，新数据中多余的列会被丢弃
    X_full_aligned = X_full_fe.reindex(columns=loaded_feature_columns, fill_value=0)
    print(f"特征工程和列对齐后 X_full_aligned 形状: {X_full_aligned.shape}")


    # --- 重新进行与训练时相同的 Train/Test Split 以获取相同的测试集 ---
    # 注意：这假设您在原始 `rv_train` 中使用了固定的 random_state 和 test_size
    # 如果您需要评估整个数据集，或者有单独保存的测试集，可以调整此部分
    _, X_test_original_aligned, _, y_test_original = train_test_split(
        X_full_aligned, y_full, test_size=0.2, random_state=42 # 使用与训练时相同的参数
    )
    print(f"分离出的测试集 X_test_original_aligned 形状: {X_test_original_aligned.shape}, y_test_original 形状: {y_test_original.shape}")


    # --- 应用加载的预处理器 ---
    print("应用加载的 imputer 和 scaler 到测试数据...")
    try:
        X_test_imputed_array = loaded_imputer.transform(X_test_original_aligned)
        X_test_imputed_df = pd.DataFrame(X_test_imputed_array, columns=loaded_feature_columns, index=X_test_original_aligned.index)

        X_test_scaled_array = loaded_scaler.transform(X_test_imputed_df)
        X_test_scaled_for_eval = pd.DataFrame(X_test_scaled_array, columns=loaded_feature_columns, index=X_test_original_aligned.index)
        print("测试数据预处理完成。")
    except Exception as e:
        print(f"应用预处理器时发生错误: {e}")
        print("请确保加载的预处理器与当前数据特征兼容。")
        # 打印一些信息帮助调试
        print(f"loaded_feature_columns (数量: {len(loaded_feature_columns)}): {loaded_feature_columns[:5]}...")
        print(f"X_test_original_aligned.columns (数量: {len(X_test_original_aligned.columns)}): {X_test_original_aligned.columns[:5]}...")
        # 检查是否有不匹配的列
        missing_in_test = set(loaded_feature_columns) - set(X_test_original_aligned.columns)
        extra_in_test = set(X_test_original_aligned.columns) - set(loaded_feature_columns)
        if missing_in_test: print(f"测试数据中缺失的列 (应由reindex补0): {missing_in_test}")
        if extra_in_test: print(f"测试数据中多余的列 (应由reindex删除): {extra_in_test}")
        # 检查NaN
        print("X_test_original_aligned 中每列的NaN统计 (在imputer之前):")
        with pd.option_context('display.max_rows', 100):
             print(X_test_original_aligned.isna().sum()[X_test_original_aligned.isna().sum() > 0])

        return


    # --- 调用评估函数 ---
    # 由于我们只加载了测试数据，所以不传递训练数据给 evaluate_model
    evaluate_model(model=loaded_model,
                   X_test_scaled_df=X_test_scaled_for_eval,
                   y_test=y_test_original,
                   feature_column_names=loaded_feature_columns,
                   X_train_scaled_df=None, #显式传递None
                   y_train=None) #显式传递None

    print("\n--- 加载和评估流程完毕 ---")

def rv_train(csv_filepath: str, target_column_name: str):
    print(f"--- 步骤 1: 从 {csv_filepath} 加载数据 ---")
    try:
        data_df = pd.read_csv(csv_filepath)
        print(f"数据加载成功。原始数据形状: {data_df.shape}")
    except FileNotFoundError:
        print(f"错误: 数据文件 {csv_filepath} 未找到。")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    if target_column_name not in data_df.columns:
        print(f"错误: 目标列 '{target_column_name}' 在CSV文件中未找到。可用列: {data_df.columns.tolist()}")
        return

    data_df[target_column_name] = data_df[target_column_name].fillna(0)
    y = data_df[target_column_name]
    X = data_df.drop(columns=[target_column_name])
    print(f"原始特征 (X) 形状: {X.shape}, 目标 (y) 形状: {y.shape}")

    print("\n--- 步骤 2: 特征工程 ---")

    # Date feature engineering
    date_cols = ['INSR_BEGIN', 'INSR_END']
    for col in date_cols:
        if col in X.columns:
            print(f"处理日期列: {col}, 当前Dtype: {X[col].dtype}")
            X[col] = pd.to_datetime(X[col], errors='coerce')
            X[f'{col}_YEAR'] = X[col].dt.year
            X[f'{col}_MONTH'] = X[col].dt.month
            X[f'{col}_DAY'] = X[col].dt.day
    if 'INSR_BEGIN' in X.columns and 'INSR_END' in X.columns and \
       pd.api.types.is_datetime64_any_dtype(X['INSR_BEGIN']) and \
       pd.api.types.is_datetime64_any_dtype(X['INSR_END']):
        X['INSURANCE_DURATION_DAYS'] = (X['INSR_END'] - X['INSR_BEGIN']).dt.days
    X = X.drop(columns=date_cols, errors='ignore')

    # Explicitly convert 'SEX' if it's object but should be numeric
    if 'SEX' in X.columns and X['SEX'].dtype == 'object':
        print(f"处理列 'SEX', 当前Dtype: {X['SEX'].dtype}")
        X['SEX'] = pd.to_numeric(X['SEX'], errors='coerce') # Coerce will turn non-numeric to NaN

    # Identify categorical features for One-Hot Encoding
    # Consider columns that are still 'object' type OR explicitly known categoricals like 'EFFECTIVE_YR'
    # even if 'EFFECTIVE_YR' might appear numeric (e.g., '08'), it represents a category.
    print("\n识别类别特征进行独热编码:")
    categorical_to_encode = []
    # Explicitly define known categoricals, even if they might appear numeric but represent categories
    known_categoricals = ['EFFECTIVE_YR', 'TYPE_VEHICLE', 'MAKE', 'USAGE']
    if 'SEX' in X.columns and X['SEX'].dtype == 'object': # If SEX conversion to numeric failed or it is better as category
        known_categoricals.append('SEX')


    for col in X.columns:
        if X[col].dtype == 'object' or col in known_categoricals:
            if col not in categorical_to_encode:
                 categorical_to_encode.append(col)

    # Print cardinality and manage high-cardinality features
    print("\n检查类别特征的基数:")
    high_cardinality_threshold = 100 # Define a threshold for "high cardinality"
    columns_to_drop_due_to_high_cardinality = []

    for col in list(categorical_to_encode): # Iterate over a copy
        if col in X.columns: # Check if column still exists
            nunique = X[col].nunique(dropna=False) # include NaNs in unique count if relevant
            print(f"列 '{col}': {nunique} 个唯一值, Dtype: {X[col].dtype}")
            if nunique > high_cardinality_threshold:
                print(f"  警告: 列 '{col}' 基数过高 ({nunique}). 考虑: ")
                print(f"    1. 合并稀有类别 (Top N / 'Other')")
                print(f"    2. 使用其他编码方法 (Target Encoding, Hashing)")
                print(f"    3. 删除此特征 (如果对模型不重要)")
                # For this run, let's decide to drop very high cardinality features if they are excessive
                if nunique > 500: # Example: drop if more than 500 unique values
                    print(f"  决策: 列 '{col}' 基数 ({nunique}) > 500, 将被删除以避免内存错误。")
                    columns_to_drop_due_to_high_cardinality.append(col)
                    categorical_to_encode.remove(col) # Don't OHE it
        else: # Column might have been dropped (e.g. if it was a date col)
             if col in categorical_to_encode:
                categorical_to_encode.remove(col)


    if columns_to_drop_due_to_high_cardinality:
        X = X.drop(columns=columns_to_drop_due_to_high_cardinality, errors='ignore')
        print(f"已删除高基数特征: {columns_to_drop_due_to_high_cardinality}")

    if categorical_to_encode:
        print(f"将进行独热编码的类别特征: {categorical_to_encode}")
        X = pd.get_dummies(X, columns=categorical_to_encode, dummy_na=False, dtype=np.uint8) # Use uint8 for memory
        print(f"独热编码后 X 的形状: {X.shape}")
    else:
        print("没有类别特征进行独热编码。")

    # Final check: ensure all columns are numeric. Coerce any remaining 'object' types.
    # This loop should ideally not find any object columns if previous steps were exhaustive.
    print("\n最终检查并转换剩余对象列 (如有):")
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  警告: 列 '{col}' (Dtype: {X[col].dtype}) 在独热编码后仍为对象类型。尝试强制转换为数值。")
            X[col] = pd.to_numeric(X[col], errors='coerce')
            if X[col].isnull().all():
                 print(f"    注意: 列 '{col}' 强制转换后全为NaN。")


    print("\n--- X 的 Dtypes (在分割、填充、缩放之前) ---")
    X.info(verbose=True, max_cols=200, show_counts=True) # Will show if any object types remain


    print("\n--- 步骤 3: 数据分割与预处理 (填充与缩放) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    final_feature_columns = X_train.columns.tolist()
    print(f"X_train 形状: {X_train.shape}, X_test 形状: {X_test.shape}")


    print("\n--- X_train 的 Dtypes (在填充、缩放之前) ---")
    X_train.info(verbose=True, max_cols=200, show_counts=True)

    # Imputation
    print("使用均值填充缺失值 (NaN)...")
    imputer = SimpleImputer(strategy='mean')
    try:
        X_train_imputed = imputer.fit_transform(X_train) # Input X_train should be all numeric here
        X_test_imputed = imputer.transform(X_test)
    except Exception as e:
        print(f"错误发生在 SimpleImputer: {e}")
        print("请检查 X_train 的 Dtypes 和内容。确保所有列都是数值型。")
        # You can print X_train.isna().sum() to see NaNs per column before imputer
        print("X_train中每列的NaN统计:")
        with pd.option_context('display.max_rows', None): # temporarily show all rows
            print(X_train.isna().sum())
        raise # re-raise the exception to stop execution

    X_train = pd.DataFrame(X_train_imputed, columns=final_feature_columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_imputed, columns=final_feature_columns, index=X_test.index)

    # Scaling
    print("进行特征缩放 (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns=final_feature_columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_array, columns=final_feature_columns, index=X_test.index)

    # ... (Rest of your training, saving, evaluation, SHAP code) ...
    print("\n--- 步骤 4: 模型训练 ---")
    rf_params = {
        'n_estimators': 100, 'max_features': 0.6, 'max_samples': 0.7,
        'max_depth': 9, 'min_samples_leaf': 800,
        'random_state': 42, 'oob_score': True, 'n_jobs': -1
    }
    if len(X_train_scaled_df) < 50000 and rf_params['min_samples_leaf'] > 50 :
        new_min_leaf = max(10, int(len(X_train_scaled_df) * 0.005))
        print(f"警告: 训练样本较少 ({len(X_train_scaled_df)}), 将 min_samples_leaf 从 {rf_params['min_samples_leaf']} 调整为 {new_min_leaf}")
        rf_params['min_samples_leaf'] = new_min_leaf

    rf_model = RandomForestRegressor(**rf_params)
    print(f"初始化 RandomForestRegressor，参数: {rf_params}")
    rf_model.fit(X_train_scaled_df, y_train)
    print("模型训练完成。")

    print("\n--- 步骤 5: 保存模型组件 ---")
    save_model_components(rf_model, scaler, imputer, final_feature_columns,
                          MODEL_FILE, SCALER_FILE, IMPUTER_FILENAME, FEATURE_COLUMNS_FILENAME)

    print("\n--- 步骤 6: 模型评估 ---")
    y_pred_train = rf_model.predict(X_train_scaled_df)
    y_pred_test = rf_model.predict(X_test_scaled_df)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    oob_score = rf_model.oob_score_ if hasattr(rf_model, 'oob_score_') and rf_model.oob_score_ else "N/A"
    print(f"训练集 MSE: {train_mse:.4f}, 测试集 MSE: {test_mse:.4f}")
    print(f"OOB Score: {oob_score if isinstance(oob_score, str) else oob_score:.4f}")

    print("\n--- 步骤 7: 特征重要性 ---")
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': final_feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("\nTop 10 特征重要性:")
    print(importance_df.head(10))

    print("\n--- 步骤 8: SHAP 模型解释 ---")
    if not X_test_scaled_df.empty:
        try:
            explainer = shap.TreeExplainer(rf_model)
            # Sample data for SHAP if X_test_scaled_df is too large
            sample_size = min(1000, len(X_test_scaled_df))
            shap_sample_data = X_test_scaled_df.sample(sample_size, random_state=42) if sample_size > 0 else X_test_scaled_df
            
            if not shap_sample_data.empty:
                print(f"为SHAP分析抽样 {len(shap_sample_data)} 条数据...")
                shap_values = explainer.shap_values(shap_sample_data)
                print("SHAP 值计算完成。")
                # shap.summary_plot(shap_values, shap_sample_data, plot_type="bar", show=True) # Uncomment to show
                # shap.summary_plot(shap_values, shap_sample_data, show=True) # Uncomment to show
                print("SHAP 图已生成（如果 matplotlib 配置为内联显示，则可能已显示）。")
            else:
                print("SHAP分析已跳过，因为测试样本为空或采样后为空。")
        except Exception as e:
            print(f"计算或绘制 SHAP 图时发生错误: {e}")
    else:
        print("SHAP分析已跳过，因为X_test_scaled_df为空。")
    print("\n--- 训练函数执行完毕 ---")


if __name__ == '__main__':
    # rv_train(DATASET_FILE, TARGET_COLUMN)
    # load_and_evaluate(DATASET_FILE, TARGET_COLUMN, MODEL_FILE, SCALER_FILE, IMPUTER_FILENAME, FEATURE_COLUMNS_FILENAME)
    pass