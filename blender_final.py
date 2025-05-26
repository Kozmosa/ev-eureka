import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize
import joblib # For saving/loading models if needed
import os
import json
from datetime import datetime, timedelta

# --- Global Configuration ---
DATASET_DIR = '.'
DATASET_FILE = os.path.join(DATASET_DIR, 'train.csv')
TEST_DATASET_FILE = os.path.join(DATASET_DIR, 'test.csv') # 新增：测试数据集文件路径
TARGET_COLUMN = 'PREMIUM'

# Columns to drop immediately after loading
COLUMNS_TO_DROP_EARLY = ['OBJECT_ID']

# For GLM
GLM_CATEGORICAL_FEATURES = [
    'SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE', 'brand',
    'battery_type_lfp', 'insurance_commercial_third_party',
    'insurance_compulsory_third_party'
]

# For RF
RF_DATE_FEATURES = ['INSR_BEGIN', 'INSR_END'] # Dates to extract components from
RF_KNOWN_CATEGORICALS = [ # Categoricals known beforehand for RF
    'SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE', 'brand'
]
RF_HIGH_CARDINALITY_THRESHOLD = 100
RF_DROP_CARDINALITY_THRESHOLD = 500

N_SPLITS_CV = 5
RANDOM_STATE = 42

# --- Model Paths ---
MODELS_DIR = 'blender_models_v3' # Updated directory
GLM_MODEL_INFO_FILE = os.path.join(MODELS_DIR, 'glm_model_info.json')
RF_MODEL_COMPONENTS = {
    'model': os.path.join(MODELS_DIR, 'rf_full_model.joblib'),
    'preprocessor': os.path.join(MODELS_DIR, 'rf_full_preprocessor.joblib'),
    'feature_columns': os.path.join(MODELS_DIR, 'rf_full_feature_columns.json')
}
BLENDER_WEIGHTS_FILE = os.path.join(MODELS_DIR, 'blender_weights.json')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True) # Ensure dataset directory exists

# --- 1. GLM Model Training and Prediction Logic ---
def preprocess_for_glm(X_df, categorical_cols, numeric_cols, fit_preprocessor=False, preprocessor=None):
    """Prepares data for GLM: OneHotEncoding for categoricals, scaling for numerics, add constant."""
    X_processed = X_df.copy()

    if fit_preprocessor:
        ct_transformers = []
        # Ensure numeric_cols and categorical_cols actually exist in X_df
        valid_numeric_cols = [col for col in numeric_cols if col in X_processed.columns]
        valid_categorical_cols = [col for col in categorical_cols if col in X_processed.columns]

        if valid_numeric_cols:
            ct_transformers.append(('num', StandardScaler(), valid_numeric_cols))
        if valid_categorical_cols:
            # Ensure categorical columns are treated as string/object for OHE
            for col in valid_categorical_cols:
                X_processed[col] = X_processed[col].astype(str)
            ct_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_categorical_cols))

        if not ct_transformers: # If no valid columns to transform (e.g. all features are dropped or none specified)
            temp_X_for_const = X_processed.copy()
            cols_to_drop_for_const = []
            for col in temp_X_for_const.columns:
                if pd.api.types.is_datetime64_any_dtype(temp_X_for_const[col]):
                    cols_to_drop_for_const.append(col)
                elif not pd.api.types.is_numeric_dtype(temp_X_for_const[col]):
                    try: # Attempt conversion for non-numeric, non-datetime
                        temp_X_for_const[col] = pd.to_numeric(temp_X_for_const[col], errors='coerce')
                    except Exception:
                        cols_to_drop_for_const.append(col) # Drop if cannot convert
            
            if cols_to_drop_for_const:
                # Use set to avoid issues if a column name is duplicated in cols_to_drop_for_const
                unique_cols_to_drop = list(set(cols_to_drop_for_const))
                print(f"GLM Preprocessing (no transformers): Dropping non-convertible/datetime columns: {unique_cols_to_drop}")
                temp_X_for_const = temp_X_for_const.drop(columns=unique_cols_to_drop)

            temp_X_for_const = temp_X_for_const.fillna(0) # Impute NaNs that might have resulted from to_numeric coercing
            
            if temp_X_for_const.empty:
                 print("Warning: GLM preprocessing (no transformers) resulted in an empty DataFrame after dropping non-numeric/datetime columns.")
                 # Return a DataFrame with just a constant column matching the original index
                 X_with_const = pd.DataFrame({'const': np.ones(len(X_df))}, index=X_df.index)
                 return X_with_const, None

            X_with_const = sm.add_constant(temp_X_for_const.astype(float), has_constant='add')
            return X_with_const, None

        # MODIFIED: Changed remainder to 'drop' to avoid passing through unhandled dtypes like datetime64
        preprocessor = ColumnTransformer(transformers=ct_transformers, remainder='drop')
        X_transformed = preprocessor.fit_transform(X_processed) # X_processed contains original columns
                                                              # 'drop' will ensure only transformed cols remain.
    elif preprocessor: # This is the transform path (using an already fitted preprocessor)
        # Ensure categorical columns are treated as string/object for OHE during transform
        original_cat_cols_in_preprocessor = []
        for name, trans_obj, cols_list in preprocessor.transformers_: # Renamed 'trans' to 'trans_obj' to avoid conflict
            if name == 'cat':
                original_cat_cols_in_preprocessor.extend(cols_list)
        
        for col in original_cat_cols_in_preprocessor:
            if col in X_processed.columns:
                 X_processed[col] = X_processed[col].astype(str)
        # The preprocessor, when transform is called, will select the columns it was fitted on
        # and apply transformations. If it was fitted with remainder='drop', it will automatically
        # handle (drop) columns not part of its original fit schema.
        X_transformed = preprocessor.transform(X_processed)
    else: # No preprocessor provided and not fitting one (raw data + constant path)
        temp_X_for_const = X_processed.copy()
        cols_to_drop_for_const = []
        for col in temp_X_for_const.columns:
            if pd.api.types.is_datetime64_any_dtype(temp_X_for_const[col]):
                cols_to_drop_for_const.append(col)
            elif not pd.api.types.is_numeric_dtype(temp_X_for_const[col]):
                 try:
                    temp_X_for_const[col] = pd.to_numeric(temp_X_for_const[col], errors='coerce')
                 except Exception:
                    cols_to_drop_for_const.append(col)
        
        if cols_to_drop_for_const:
            unique_cols_to_drop = list(set(cols_to_drop_for_const))
            print(f"GLM Preprocessing (raw data path): Dropping non-convertible/datetime columns: {unique_cols_to_drop}")
            temp_X_for_const = temp_X_for_const.drop(columns=unique_cols_to_drop)

        temp_X_for_const = temp_X_for_const.fillna(0)
        
        if temp_X_for_const.empty:
            print("Warning: GLM preprocessing (raw data path) resulted in an empty DataFrame after dropping non-numeric/datetime columns.")
            X_with_const = pd.DataFrame({'const': np.ones(len(X_df))}, index=X_df.index)
            return X_with_const, None

        X_with_const = sm.add_constant(temp_X_for_const.astype(float), has_constant='add')
        return X_with_const, None

    # Feature names extraction
    try:
        # For scikit-learn >= 0.24 (approx)
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError: # Fallback for older scikit-learn versions
        feature_names = []
        for name, trans_obj, cols_list_orig in preprocessor.transformers_: # Renamed 'trans' to 'trans_obj'
            # Make sure cols_list_orig contains strings for get_feature_names if they are column names
            cols_list_for_get_names = [str(c) for c in cols_list_orig]

            if hasattr(trans_obj, 'get_feature_names_out'): # For transformers like OneHotEncoder in newer versions
                 # Pass input_features if the transformer expects it (like OHE)
                try:
                    feature_names.extend(trans_obj.get_feature_names_out(cols_list_for_get_names))
                except TypeError: # Some versions of get_feature_names_out don't take args
                    feature_names.extend(trans_obj.get_feature_names_out())

            elif hasattr(trans_obj, 'get_feature_names') and name == 'cat': # For older OneHotEncoder
                # OHE get_feature_names needs input_features=cols_list_for_get_names
                ohe_feature_names = trans_obj.get_feature_names(input_features=cols_list_for_get_names)
                feature_names.extend(ohe_feature_names)
            elif name == 'num': # For StandardScaler or other numeric transformers
                feature_names.extend(cols_list_orig) # Original numeric column names are preserved
            # No need to handle 'passthrough' for remainder if remainder='drop'
            # If remainder was 'passthrough', additional logic would be needed here for older sklearn.
            # elif trans_obj == 'passthrough':
            # feature_names.extend(cols_list_orig)

        # This part for remainder='passthrough' is not strictly needed if remainder='drop',
        # but kept for context if 'passthrough' was ever re-enabled.
        # if preprocessor.remainder == 'passthrough': # This block would only run if remainder was 'passthrough'
        #     # Get columns that were neither in 'num' nor 'cat'
        #     # This logic for older sklearn with passthrough can be complex.
        #     # For remainder='drop', this section is not hit for the remainder part.
        #     pass


    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_processed.index)
    X_transformed_df = sm.add_constant(X_transformed_df.astype(float), has_constant='add') # Ensure float before adding const
    return X_transformed_df, preprocessor

def train_glm_on_fold(X_train_df, y_train_series, glm_categorical_features, glm_numeric_features):
    """Trains a GLM model on a fold. Uses Tweedie with log link."""
    X_train_processed, preprocessor = preprocess_for_glm(
        X_train_df,
        glm_categorical_features,
        glm_numeric_features,
        fit_preprocessor=True
    )
    y_train_series = y_train_series.astype(float)
    y_train_series = np.maximum(y_train_series, 0.001) # Ensure positive for log link

    if X_train_processed.shape[1] == 1 and 'const' in X_train_processed.columns: # Only const column left
        print("Warning: GLM training data has no features other than constant. Skipping GLM training for this fold.")
        # Return dummy results or handle as an error
        class DummyGLMResults:
            def __init__(self, X_cols):
                self.params = pd.Series([0.0] * X_cols.shape[1], index=X_cols.columns) # Predict 0
            def predict(self, X):
                return np.zeros(len(X))
        
        # Create dummy preprocessor info if preprocessor is None (e.g. from "no transformers" path)
        if preprocessor is None:
            class DummyPreprocessor:
                def __init__(self):
                    self.transformers_ = [] # empty transformers list
                def transform(self, X): # basic passthrough or empty
                    return X 
                def get_feature_names_out(self, *args): # basic passthrough or empty
                    return X.columns.tolist()
            preprocessor = DummyPreprocessor()

        return DummyGLMResults(X_train_processed), preprocessor, X_train_processed.columns.tolist()


    try:
        glm_model_obj = sm.GLM(y_train_series, X_train_processed.astype(float), # Ensure float for GLM
                               family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5))
        glm_results = glm_model_obj.fit()
        return glm_results, preprocessor, X_train_processed.columns.tolist()
    except Exception as e:
        print(f"Error training GLM: {e}")
        print("X_train_processed dtypes:\n", X_train_processed.dtypes)
        print("X_train_processed NaNs per column:\n", X_train_processed.isnull().sum()[X_train_processed.isnull().sum() > 0])
        print("y_train_series NaNs:", y_train_series.isnull().sum())
        print("y_train_series non-positive:", (y_train_series <=0).sum())
        if hasattr(e, 'summary'): print(e.summary())
        raise

def predict_glm_on_fold(glm_results, X_val_df, preprocessor, feature_columns_glm):
    """Predicts using a trained GLM model on validation data."""
    X_val_processed, _ = preprocess_for_glm(
        X_val_df, [], [], fit_preprocessor=False, preprocessor=preprocessor
    )
    
    # Align columns to match training
    # Ensure X_val_aligned has all columns from feature_columns_glm, in the correct order, filling missing with 0
    X_val_aligned = pd.DataFrame(0.0, index=X_val_processed.index, columns=feature_columns_glm)
    common_cols = X_val_aligned.columns.intersection(X_val_processed.columns)
    X_val_aligned[common_cols] = X_val_processed[common_cols]
    
    if 'const' in feature_columns_glm: # Ensure const column is 1.0 if it's part of the model
        X_val_aligned['const'] = 1.0
    
    # Ensure order of columns matches feature_columns_glm for prediction
    X_val_aligned = X_val_aligned[feature_columns_glm] 
    
    return glm_results.predict(X_val_aligned.astype(float)) # Ensure float for GLM


# --- 2. Random Forest Model Training and Prediction Logic ---
def create_rf_preprocessor(X_df_input,
                           date_features, # RF_DATE_FEATURES
                           known_categoricals, # RF_KNOWN_CATEGORICALS
                           high_card_thresh, # RF_HIGH_CARDINALITY_THRESHOLD
                           drop_card_thresh, # RF_DROP_CARDINALITY_THRESHOLD
                           fit_preprocessor_flag=False, # True for train, False for predict (pass fitted preprocessor)
                           fitted_preprocessor_ct=None): # Pass the fitted ColumnTransformer for prediction
    """Creates or applies a preprocessor pipeline for RF."""
    X_transformed_df = X_df_input.copy()

    # 1. Date features: extract components
    for col in date_features:
        if col in X_transformed_df.columns:
            # Convert to datetime, coercing errors. Original script had this, good for robustness.
            X_transformed_df[col] = pd.to_datetime(X_transformed_df[col], errors='coerce')
            X_transformed_df[f'{col}_YEAR'] = X_transformed_df[col].dt.year
            X_transformed_df[f'{col}_MONTH'] = X_transformed_df[col].dt.month
            X_transformed_df[f'{col}_DAY'] = X_transformed_df[col].dt.day
            X_transformed_df[f'{col}_DAYOFWEEK'] = X_transformed_df[col].dt.dayofweek
            X_transformed_df[f'{col}_WEEKOFYEAR'] = X_transformed_df[col].dt.isocalendar().week.astype(float).fillna(0).astype(int) # Ensure int
    # Drop original date columns after extraction
    X_transformed_df = X_transformed_df.drop(columns=[col for col in date_features if col in X_transformed_df.columns], errors='ignore')

    if fit_preprocessor_flag:
        # 2. Identify categorical and numerical features for fitting preprocessor
        categorical_features_identified = []
        numeric_features_identified = []

        for col in X_transformed_df.columns:
            if col in known_categoricals or X_transformed_df[col].dtype == 'object':
                X_transformed_df[col] = X_transformed_df[col].astype(str) # Ensure string type for consistency
                nunique = X_transformed_df[col].nunique()
                if nunique > drop_card_thresh:
                    print(f"RF Preprocessing: Dropping '{col}' due to very high cardinality ({nunique}).")
                    # We don't drop from X_transformed_df here, ColumnTransformer will handle it with remainder='drop'
                    # or we can explicitly exclude it from features list.
                    # For now, let's exclude it from categorical_features_identified.
                    continue
                elif nunique > 0: # Only add if there are actual categories (not all NaN after astype(str))
                    categorical_features_identified.append(col)
            elif pd.api.types.is_numeric_dtype(X_transformed_df[col]):
                numeric_features_identified.append(col)
            # else: other types will be dropped by ColumnTransformer's remainder='drop'

        # Ensure no overlap and features exist
        categorical_features_identified = [c for c in categorical_features_identified if c in X_transformed_df.columns]
        numeric_features_identified = [n for n in numeric_features_identified if n in X_transformed_df.columns and n not in categorical_features_identified]

        # Define transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Impute with most frequent for string categoricals
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        transformers_list = []
        if numeric_features_identified:
            transformers_list.append(('num', numeric_transformer, numeric_features_identified))
        if categorical_features_identified:
            transformers_list.append(('cat', categorical_transformer, categorical_features_identified))

        if not transformers_list: # No features to transform
             # Return an empty DataFrame with original index if no features are processed
            print("Warning: RF preprocessor found no numeric or categorical features to transform.")
            return pd.DataFrame(index=X_df_input.index), None, []


        preprocessor_ct = ColumnTransformer(transformers=transformers_list, remainder='drop')
        X_final_array = preprocessor_ct.fit_transform(X_transformed_df)
        final_columns_out = preprocessor_ct.get_feature_names_out()
        X_final_df = pd.DataFrame(X_final_array, columns=final_columns_out, index=X_df_input.index)
        return X_final_df, preprocessor_ct, final_columns_out

    elif fitted_preprocessor_ct: # Apply a pre-fitted ColumnTransformer
        # Ensure categorical columns are str for the transformer
        # This relies on `fitted_preprocessor_ct.transformers_` to know which were categorical
        for name, trans_pipeline, cat_cols_list in fitted_preprocessor_ct.transformers_:
            if name == 'cat': # Check if it's the categorical transformer
                # trans_pipeline is a Pipeline object, get the OHE's original columns
                # The cat_cols_list are the original column names fed to this categorical pipeline
                for cat_col in cat_cols_list:
                    if cat_col in X_transformed_df.columns:
                        X_transformed_df[cat_col] = X_transformed_df[cat_col].astype(str)
        
        X_final_array = fitted_preprocessor_ct.transform(X_transformed_df)
        final_columns_out = fitted_preprocessor_ct.get_feature_names_out()
        X_final_df = pd.DataFrame(X_final_array, columns=final_columns_out, index=X_df_input.index)
        return X_final_df, fitted_preprocessor_ct, final_columns_out # Return the same preprocessor
    else:
        raise ValueError("Either fit_preprocessor_flag must be True, or a fitted_preprocessor_ct must be provided.")


def train_rf_on_fold(X_train_df, y_train_series):
    """Trains a Random Forest model on a fold."""
    X_train_processed, preprocessor_ct, feature_cols = create_rf_preprocessor(
        X_train_df,
        RF_DATE_FEATURES, RF_KNOWN_CATEGORICALS,
        RF_HIGH_CARDINALITY_THRESHOLD, RF_DROP_CARDINALITY_THRESHOLD,
        fit_preprocessor_flag=True # Fit the preprocessor
    )
    y_train_series = y_train_series.astype(float)

    if X_train_processed.empty or X_train_processed.shape[1] == 0:
        print("Warning: RF training data is empty or has no features after preprocessing. Skipping RF training for this fold.")
        # Return a dummy model and the preprocessor (even if it results in no features)
        # Predictions from this dummy model should be handled (e.g., default to mean or zero)
        class DummyRF:
            def predict(self, X): return np.zeros(len(X)) # Predict zeros
            def fit(self, X, y): return self # Dummy fit
        
        # If preprocessor_ct is None (e.g., from create_rf_preprocessor if no features found), create a dummy
        if preprocessor_ct is None:
            class DummyPreprocessor:
                def transform(self, X): return pd.DataFrame(index=X.index) # Return empty DF
                def get_feature_names_out(self): return []
            preprocessor_ct = DummyPreprocessor()

        return DummyRF(), preprocessor_ct, []


    rf_params = {
        'n_estimators': 100, 'max_features': 0.6, 'max_samples': 0.7,
        'max_depth': 9, 'min_samples_leaf': max(10, int(len(X_train_df) * 0.005)),
        'random_state': RANDOM_STATE, 'oob_score': False, 'n_jobs': -1
    }
    rf_model_obj = RandomForestRegressor(**rf_params)
    rf_model_obj.fit(X_train_processed, y_train_series)
    return rf_model_obj, preprocessor_ct, feature_cols

def predict_rf_on_fold(rf_model, X_val_df, fitted_preprocessor_ct):
    """Predicts using a trained RF model on validation data with its preprocessor."""
    # The create_rf_preprocessor function now handles both date transformation and ColumnTransformer application
    X_val_processed, _, _ = create_rf_preprocessor(
        X_val_df,
        RF_DATE_FEATURES, RF_KNOWN_CATEGORICALS, # These are for reference, not for re-fitting
        RF_HIGH_CARDINALITY_THRESHOLD, RF_DROP_CARDINALITY_THRESHOLD, # Not used when applying
        fit_preprocessor_flag=False,
        fitted_preprocessor_ct=fitted_preprocessor_ct
    )
    if X_val_processed.empty or X_val_processed.shape[1] == 0:
        # This can happen if fitted_preprocessor_ct was a dummy or resulted in no features
        print("Warning: RF validation data is empty or has no features after preprocessing. Returning zeros.")
        return np.zeros(len(X_val_df))
        
    return rf_model.predict(X_val_processed)


# --- 3. Blending Weights Optimization ---
def loss_function(weights, pred_glm, pred_rf, y_true):
    """MSE loss for blending weights."""
    w_glm, w_rf = weights
    blended_pred = w_glm * pred_glm + w_rf * pred_rf
    return mean_squared_error(y_true, blended_pred)

def optimize_blending_weights(pred_glm_oof, pred_rf_oof, y_true_oof):
    """Finds optimal weights w_glm, w_rf."""
    constraints = ({'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1},
                   {'type': 'ineq', 'fun': lambda weights: weights[0]}, # w_glm >= 0
                   {'type': 'ineq', 'fun': lambda weights: weights[1]})  # w_rf >= 0
    initial_weights = [0.5, 0.5]
    bounds = [(0, 1), (0, 1)]

    result = minimize(loss_function, initial_weights,
                      args=(pred_glm_oof, pred_rf_oof, y_true_oof),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    if result.success:
        return result.x
    else:
        print(f"Warning: Weight optimization failed. Reason: {result.message}. Returning initial weights.")
        return initial_weights


# --- Main Orchestration ---
def main():
    print(f"Loading data from: {DATASET_FILE}")
    try:
        data_df = pd.read_csv(DATASET_FILE)
        if data_df.columns[0].startswith('Unnamed: '):
            data_df = data_df.drop(columns=[data_df.columns[0]])
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        print("Creating a dummy dataset for demonstration purposes based on new schema...")
        num_samples = 500
        
        # Generate date strings
        start_date = datetime(2020, 1, 1)
        insr_begin_dates = [(start_date + timedelta(days=np.random.randint(0, 1000))).strftime('%d-%b-%y').upper() for _ in range(num_samples)]
        insr_end_dates = [(pd.to_datetime(d, format='%d-%b-%y') + timedelta(days=365)).strftime('%d-%b-%y').upper() for d in insr_begin_dates]

        data_df = pd.DataFrame({
            'SEX': np.random.choice([1, 2, 9], num_samples, p=[0.49, 0.49, 0.02]), # Example codes for SEX
            'INSR_BEGIN': insr_begin_dates,
            'INSR_END': insr_end_dates,
            'EFFECTIVE_YR': np.random.choice([2020, 2021, 2022, 2023], num_samples),
            'INSR_TYPE': np.random.choice([1206, 1207, 1201], num_samples), # Example insurance type codes
            'INSURED_VALUE': np.random.uniform(50000, 200000, num_samples),
            'PREMIUM': np.random.uniform(500, 5000, num_samples),
            'OBJECT_ID': [f'ID_{i}' for i in range(num_samples)], # Will be dropped
            'PROD_YEAR': np.random.randint(2010, 2023, num_samples),
            'SEATS_NUM': np.random.choice([2, 4, 5, 7], num_samples),
            'CARRYING_CAPACITY': np.random.choice([0, 500, 1000, 1500], num_samples) * (np.random.rand(num_samples) < 0.3), # Often 0 for cars
            'TYPE_VEHICLE': np.random.choice(['轿车', 'SUV', 'MPV', '货车'], num_samples),
            'CCM_TON': np.random.uniform(1000, 5000, num_samples), # Engine CC or Tonnage
            'MAKE': np.random.choice(['宝马', '奔驰', '奥迪', '丰田', '本田', '吉利'], num_samples),
            'USAGE': np.random.choice(['家用', '营运', '租赁'], num_samples),
            'CLAIM_PAID': np.random.gamma(1, scale=5000, size=num_samples) * (np.random.rand(num_samples) < 0.2), # Some policies have claims
            'brand': np.random.choice(['品牌A', '品牌B', '品牌C', '品牌D'], num_samples),
            'average_speed': np.random.uniform(20, 80, num_samples),
            'avg_daily_charges': np.random.poisson(1, num_samples),
            'fatigue_driving_ratio': np.random.uniform(0, 0.1, num_samples),
            'late_night_trip_ratio': np.random.uniform(0, 0.2, num_samples),
            'avg_late_night_trip_mileage': np.random.uniform(0, 50, num_samples) * (np.random.rand(num_samples) < 0.3),
            'high_temp_driving_ratio': np.random.uniform(0, 0.3, num_samples),
            'battery_type_lfp': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
            'initial_battery_soc': np.random.uniform(20, 100, num_samples),
            'avg_charge_duration': np.random.uniform(3000, 30000, num_samples),
            'insurance_commercial_third_party': np.random.choice([0, 1], num_samples, p=[0.4, 0.6]),
            'insurance_compulsory_third_party': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
            TARGET_COLUMN: np.random.gamma(2, scale=1000, size=num_samples) + np.random.rand(num_samples) * 500
        })
        data_df[TARGET_COLUMN] = np.maximum(0.01, data_df[TARGET_COLUMN])
        # Save dummy data if created
        try:
            data_df.to_csv(DATASET_FILE, index=False)
            print(f"Dummy dataset saved to {DATASET_FILE}")
        except Exception as e_save:
            print(f"Could not save dummy dataset: {e_save}")


    if TARGET_COLUMN not in data_df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return

    # Drop specified columns early
    cols_to_drop_existing = [col for col in COLUMNS_TO_DROP_EARLY if col in data_df.columns]
    if cols_to_drop_existing:
        print(f"Dropping early: {cols_to_drop_existing}")
        data_df = data_df.drop(columns=cols_to_drop_existing)

    data_df[TARGET_COLUMN] = data_df[TARGET_COLUMN].fillna(data_df[TARGET_COLUMN].median())
    y = data_df[TARGET_COLUMN]
    X = data_df.drop(columns=[TARGET_COLUMN])

    # Convert date columns to datetime objects for consistent processing if they are not already
    # This is important if they are read as strings from CSV
    for date_col in RF_DATE_FEATURES:
        if date_col in X.columns:
            try:
                X[date_col] = pd.to_datetime(X[date_col], errors='coerce') # Coerce errors to NaT
            except Exception as e_dt:
                print(f"Warning: Could not convert column {date_col} to datetime: {e_dt}. It will be NaT.")
                X[date_col] = pd.NaT


    # Determine GLM numeric features
    all_feature_cols = X.columns.tolist()
    valid_glm_categorical_features = [col for col in GLM_CATEGORICAL_FEATURES if col in X.columns]
    
    # GLM numeric features are those not in date features or categorical features, and are numeric
    potential_numeric_glm = [
        col for col in all_feature_cols
        if col not in valid_glm_categorical_features and col not in RF_DATE_FEATURES # Exclude dates for GLM direct input
    ]
    glm_numeric_features = []
    for col in potential_numeric_glm:
        # Check if column is numeric after potential date conversions
        if pd.api.types.is_numeric_dtype(X[col]):
            glm_numeric_features.append(col)
        # If it was a date column that's now datetime64, it won't be added unless converted to numeric (e.g. timestamp)
        # For now, RF_DATE_FEATURES are explicitly excluded for GLM direct input.

    print(f"GLM Categorical Features used: {valid_glm_categorical_features}")
    print(f"GLM Numeric Features used (first 5): {glm_numeric_features[:5]}... (Total: {len(glm_numeric_features)})")
    print(f"RF Date Features to be processed: {RF_DATE_FEATURES}")
    print(f"RF Known Categorical Features: {RF_KNOWN_CATEGORICALS}")


    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_glm = np.zeros(len(X))
    oof_preds_rf = np.zeros(len(X))
    oof_true_y = np.zeros(len(X)) # Store true y for OOF MSE calculation

    print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation for OOF predictions...")
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold_idx + 1}/{N_SPLITS_CV} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print("Training GLM...")
        glm_model_fold, glm_preprocessor_fold, glm_cols_fold = train_glm_on_fold(
            X_train.copy(), y_train, valid_glm_categorical_features, glm_numeric_features
        )
        
        # Check if glm_model_fold is a dummy (from no features)
        if hasattr(glm_model_fold, 'predict') and len(glm_cols_fold) > 0 : # Basic check for a valid model
            oof_preds_glm[val_index] = predict_glm_on_fold(glm_model_fold, X_val.copy(), glm_preprocessor_fold, glm_cols_fold)
        else:
            print("GLM for fold resulted in no features or dummy model. Predictions will be 0.")
            oof_preds_glm[val_index] = 0.0 # Default prediction if GLM failed or was dummy

        print("GLM OOF predictions for fold complete.")

        print("Training RF...")
        rf_model_fold, rf_preprocessor_fold, rf_feature_cols_fold = train_rf_on_fold(X_train.copy(), y_train)
        if isinstance(rf_model_fold, RandomForestRegressor) and len(list(rf_feature_cols_fold)) > 0: # Check if actual model was trained & has features
            oof_preds_rf[val_index] = predict_rf_on_fold(rf_model_fold, X_val.copy(), rf_preprocessor_fold)
        else: # Dummy model was returned or no features
            print("RF for fold resulted in no features or dummy model. Predictions will be 0.")
            oof_preds_rf[val_index] = 0.0 # Default prediction
        print("RF OOF predictions for fold complete.")

        oof_true_y[val_index] = y_val # Store actual y values for this fold

    # Handle potential NaNs in OOF predictions (e.g., if a model failed for a fold or a default value was used)
    # It's better to ensure models predict valid numbers, but this is a fallback.
    if np.sum(np.isnan(oof_preds_glm)) > 0: # Check if any NaNs exist
        glm_mean_oof = np.nanmean(oof_preds_glm)
        if np.isnan(glm_mean_oof): glm_mean_oof = 0 # If all are NaN, use 0
        print(f"Warning: {np.sum(np.isnan(oof_preds_glm))} NaNs found in GLM OOF predictions. Imputing with mean ({glm_mean_oof:.4f}).")
        oof_preds_glm = np.nan_to_num(oof_preds_glm, nan=glm_mean_oof)
    if np.sum(np.isnan(oof_preds_rf)) > 0:
        rf_mean_oof = np.nanmean(oof_preds_rf)
        if np.isnan(rf_mean_oof): rf_mean_oof = 0
        print(f"Warning: {np.sum(np.isnan(oof_preds_rf))} NaNs found in RF OOF predictions. Imputing with mean ({rf_mean_oof:.4f}).")
        oof_preds_rf = np.nan_to_num(oof_preds_rf, nan=rf_mean_oof)


    print("\n--- OOF Predictions Collected ---")
    mse_glm_oof = mean_squared_error(oof_true_y, oof_preds_glm)
    mse_rf_oof = mean_squared_error(oof_true_y, oof_preds_rf)
    print(f"OOF MSE GLM: {mse_glm_oof:.4f}")
    print(f"OOF MSE RF: {mse_rf_oof:.4f}")

    print("\n--- Optimizing Blender Weights ---")
    blender_weights = optimize_blending_weights(oof_preds_glm, oof_preds_rf, oof_true_y)
    w_glm, w_rf = blender_weights
    print(f"Optimized Weights: GLM = {w_glm:.4f}, RF = {w_rf:.4f}")

    blended_oof_predictions = w_glm * oof_preds_glm + w_rf * oof_preds_rf
    mse_blended_oof = mean_squared_error(oof_true_y, blended_oof_predictions)
    print(f"OOF MSE Blended: {mse_blended_oof:.4f}")

    with open(BLENDER_WEIGHTS_FILE, 'w') as f:
        json.dump({'w_glm': w_glm, 'w_rf': w_rf, 'mse_blended_oof': mse_blended_oof}, f)
    print(f"Blender weights saved to {BLENDER_WEIGHTS_FILE}")

    print("\n--- Training Final Models on Full Dataset ---")
    print("Training final GLM model...")
    final_glm_model, final_glm_preprocessor, final_glm_cols = train_glm_on_fold(
        X.copy(), y, valid_glm_categorical_features, glm_numeric_features
    )
    # Check if final_glm_model is valid before saving
    if hasattr(final_glm_model, 'params') and final_glm_preprocessor is not None and len(final_glm_cols) > 0:
        joblib.dump(final_glm_preprocessor, os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib'))
        joblib.dump(final_glm_model.params.to_dict(), os.path.join(MODELS_DIR, 'final_glm_model_params.json'))
        with open(GLM_MODEL_INFO_FILE, 'w') as f:
            json.dump({
                'columns_fitted': final_glm_cols, 
                'categorical_features_input': valid_glm_categorical_features, 
                'numeric_features_input': glm_numeric_features, 
                'family_info': {'class': 'Tweedie', 'link': 'Log', 'var_power': 1.5}
                }, f)
        print("Final GLM model (components) saved.")
    else:
        print("Final GLM model training resulted in a dummy or invalid model. Not saving GLM components.")


    print("Training final RF model...")
    final_rf_model, final_rf_preprocessor_ct, final_rf_cols = train_rf_on_fold(X.copy(), y)
    if isinstance(final_rf_model, RandomForestRegressor) and final_rf_preprocessor_ct is not None and  len(list(final_rf_cols)) > 0 :
        joblib.dump(final_rf_model, RF_MODEL_COMPONENTS['model'])
        joblib.dump(final_rf_preprocessor_ct, RF_MODEL_COMPONENTS['preprocessor']) 
        with open(RF_MODEL_COMPONENTS['feature_columns'], 'w') as f:
            json.dump(list(final_rf_cols), f) 
        print("Final RF model and preprocessor saved.")
    else:
        print("Final RF model training resulted in a dummy or invalid model. Not saving RF components.")

    print("\n--- Blending Process Complete ---")


def predict_on_new_data(new_X_df_input):
    print("\n--- Predicting on New Data (Batch) ---")
    # Check for essential model files
    glm_preprocessor_file = os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib')
    glm_params_file = os.path.join(MODELS_DIR, 'final_glm_model_params.json')

    glm_components_exist = all(os.path.exists(f) for f in [
        BLENDER_WEIGHTS_FILE, glm_preprocessor_file, glm_params_file, GLM_MODEL_INFO_FILE
    ])
    
    rf_components_exist = all(os.path.exists(RF_MODEL_COMPONENTS[key]) for key in ['model', 'preprocessor', 'feature_columns'])


    if not glm_components_exist:
        print("Error: Not all GLM model components found. Run main training first. Cannot make predictions.")
        return None # Cannot proceed without GLM components for blending weights at least
    
    with open(BLENDER_WEIGHTS_FILE, 'r') as f:
        blender_info = json.load(f)
    w_glm, w_rf = blender_info['w_glm'], blender_info['w_rf']
    print(f"Loaded blender weights: GLM={w_glm:.4f}, RF={w_rf:.4f}")

    new_X_df = new_X_df_input.copy()
    # Drop the same columns that were dropped during training
    cols_to_drop_existing_new = [col for col in COLUMNS_TO_DROP_EARLY if col in new_X_df.columns]
    if cols_to_drop_existing_new:
        new_X_df = new_X_df.drop(columns=cols_to_drop_existing_new)

    # Convert date columns to datetime objects for consistent processing
    for date_col in RF_DATE_FEATURES:
        if date_col in new_X_df.columns:
            try:
                new_X_df[date_col] = pd.to_datetime(new_X_df[date_col], errors='coerce')
            except Exception as e_dt_new:
                print(f"Warning: Could not convert column {date_col} in new data to datetime: {e_dt_new}. It will be NaT.")
                new_X_df[date_col] = pd.NaT


    # --- GLM Prediction on New Data ---
    print("Preparing GLM prediction...")
    pred_glm_new = np.zeros(len(new_X_df)) # Default to zeros
    try:
        glm_preprocessor_loaded = joblib.load(glm_preprocessor_file)
        glm_params_loaded = pd.Series(joblib.load(glm_params_file))
        with open(GLM_MODEL_INFO_FILE, 'r') as f:
            glm_info = json.load(f)
        glm_cols_fitted_on = glm_info['columns_fitted'] 

        new_X_glm_processed, _ = preprocess_for_glm(
            new_X_df.copy(), [], [], fit_preprocessor=False, preprocessor=glm_preprocessor_loaded
        )

        X_glm_aligned = pd.DataFrame(0.0, index=new_X_glm_processed.index, columns=glm_cols_fitted_on)
        common_cols = X_glm_aligned.columns.intersection(new_X_glm_processed.columns)
        X_glm_aligned[common_cols] = new_X_glm_processed[common_cols]
        if 'const' in glm_cols_fitted_on: X_glm_aligned['const'] = 1.0
        X_glm_aligned = X_glm_aligned[glm_cols_fitted_on] 

        family_info = glm_info.get('family_info', {'class': 'Tweedie', 'link': 'Log', 'var_power': 1.5})
        link_func_str = family_info.get('link', 'Log').lower()
        link_func = sm.families.links.Log() 
        if link_func_str == 'identity': link_func = sm.families.links.identity()
        
        glm_family_class_str = family_info.get('class', 'Tweedie').lower()
        var_power = family_info.get('var_power', 1.5)
        if glm_family_class_str == 'tweedie': glm_family = sm.families.Tweedie(link=link_func, var_power=var_power)
        elif glm_family_class_str == 'gamma': glm_family = sm.families.Gamma(link=link_func)
        elif glm_family_class_str == 'poisson': glm_family = sm.families.Poisson(link=link_func)
        else: glm_family = sm.families.Tweedie(link=link_func, var_power=var_power) 

        dummy_exog = pd.DataFrame(np.ones((1, len(glm_cols_fitted_on))), columns=glm_cols_fitted_on)
        if 'const' not in dummy_exog.columns and 'const' in glm_cols_fitted_on:
             dummy_exog['const'] = 1.0 
        dummy_endog = pd.Series([1.0]) 

        dummy_glm_model = sm.GLM(endog=dummy_endog, exog=dummy_exog[glm_cols_fitted_on], family=glm_family)
        
        pred_glm_new = dummy_glm_model.predict(params=glm_params_loaded, exog=X_glm_aligned.astype(float))
        print("GLM prediction on new data successful (using loaded params).")
    except Exception as e:
        print(f"Error making GLM prediction with loaded params: {e}. GLM predictions will be zero.")
        # pred_glm_new remains zeros

    # --- RF Prediction on New Data ---
    print("Preparing RF prediction...")
    pred_rf_new = np.zeros(len(new_X_df)) # Default to zeros
    if rf_components_exist:
        try:
            rf_model_loaded = joblib.load(RF_MODEL_COMPONENTS['model'])
            rf_preprocessor_ct_loaded = joblib.load(RF_MODEL_COMPONENTS['preprocessor'])
            
            new_X_rf_processed, _, _ = create_rf_preprocessor(
                new_X_df.copy(), 
                RF_DATE_FEATURES, RF_KNOWN_CATEGORICALS, 
                RF_HIGH_CARDINALITY_THRESHOLD, RF_DROP_CARDINALITY_THRESHOLD, 
                fit_preprocessor_flag=False,
                fitted_preprocessor_ct=rf_preprocessor_ct_loaded
            )
            
            if new_X_rf_processed.empty or new_X_rf_processed.shape[1] == 0:
                 print("Warning: RF new data is empty or has no features after preprocessing. RF predictions will be zero.")
            else:
                pred_rf_new = rf_model_loaded.predict(new_X_rf_processed)
                print("RF prediction on new data successful.")
        except Exception as e:
            print(f"Error making RF prediction: {e}. RF predictions will be zero.")
    else:
        print("RF model components not found or incomplete. RF predictions will be zero.")


    blended_predictions_new = w_glm * pred_glm_new + w_rf * pred_rf_new
    print("Blending complete for new data.")
    return blended_predictions_new


def predict_single_instance_blended(single_instance_dict: dict) -> float:
    """
    Loads pre-trained blended model components and predicts on a single new data instance.
    """
    print("--- Predicting on Single Instance ---")

    # Check for model files
    glm_preprocessor_file = os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib')
    glm_params_file = os.path.join(MODELS_DIR, 'final_glm_model_params.json')
    
    required_glm_files = [ BLENDER_WEIGHTS_FILE, glm_preprocessor_file, glm_params_file, GLM_MODEL_INFO_FILE ]
    
    rf_model_file = RF_MODEL_COMPONENTS['model']
    rf_preprocessor_file = RF_MODEL_COMPONENTS['preprocessor']

    if not all(os.path.exists(f) for f in required_glm_files):
        print(f"Error: Not all required GLM/Blender model components found. Cannot predict.")
        # Identify which specific file is missing
        for f_path in required_glm_files:
            if not os.path.exists(f_path):
                print(f"Missing component: {f_path}")
        return np.nan
    
    rf_components_exist = os.path.exists(rf_model_file) and os.path.exists(rf_preprocessor_file)
    if not rf_components_exist:
        print("Warning: RF model components not found. RF contribution to blend will be 0.")


    # Load Blender Weights
    try:
        with open(BLENDER_WEIGHTS_FILE, 'r') as f:
            blender_info = json.load(f)
        w_glm, w_rf = blender_info['w_glm'], blender_info['w_rf']
        print(f"Loaded blender weights: GLM={w_glm:.4f}, RF={w_rf:.4f}")
    except Exception as e:
        print(f"Error loading blender weights: {e}")
        return np.nan

    new_X_df = pd.DataFrame(single_instance_dict, index=[0])
    # Drop the same columns that were dropped during training
    cols_to_drop_existing_single = [col for col in COLUMNS_TO_DROP_EARLY if col in new_X_df.columns]
    if cols_to_drop_existing_single:
        new_X_df = new_X_df.drop(columns=cols_to_drop_existing_single)

    # Convert date columns to datetime objects for consistent processing
    for date_col in RF_DATE_FEATURES:
        if date_col in new_X_df.columns:
            try:
                new_X_df[date_col] = pd.to_datetime(new_X_df[date_col], errors='coerce')
            except Exception as e_dt_single:
                print(f"Warning: Could not convert column {date_col} in single instance to datetime: {e_dt_single}. It will be NaT.")
                new_X_df[date_col] = pd.NaT


    # --- GLM Prediction ---
    pred_glm_val = 0.0 # Default
    try:
        glm_preprocessor_loaded = joblib.load(glm_preprocessor_file)
        glm_params_loaded = pd.Series(joblib.load(glm_params_file))
        with open(GLM_MODEL_INFO_FILE, 'r') as f:
            glm_info = json.load(f)
        glm_cols_fitted_on = glm_info['columns_fitted']

        new_X_glm_processed, _ = preprocess_for_glm(
            new_X_df.copy(), [], [], fit_preprocessor=False, preprocessor=glm_preprocessor_loaded
        )
        X_glm_aligned = pd.DataFrame(0.0, index=new_X_glm_processed.index, columns=glm_cols_fitted_on)
        common_cols = X_glm_aligned.columns.intersection(new_X_glm_processed.columns)
        X_glm_aligned[common_cols] = new_X_glm_processed[common_cols]
        if 'const' in glm_cols_fitted_on: X_glm_aligned['const'] = 1.0
        X_glm_aligned = X_glm_aligned[glm_cols_fitted_on] 

        family_info = glm_info.get('family_info', {'class': 'Tweedie', 'link': 'Log', 'var_power': 1.5})
        link_func_str = family_info.get('link', 'Log').lower()
        link_func = sm.families.links.Log()
        if link_func_str == 'identity': link_func = sm.families.links.identity()
        
        glm_family_class_str = family_info.get('class', 'Tweedie').lower()
        var_power = family_info.get('var_power', 1.5)
        if glm_family_class_str == 'tweedie': glm_family = sm.families.Tweedie(link=link_func, var_power=var_power)
        elif glm_family_class_str == 'gamma': glm_family = sm.families.Gamma(link=link_func)
        elif glm_family_class_str == 'poisson': glm_family = sm.families.Poisson(link=link_func)
        else: glm_family = sm.families.Tweedie(link=link_func, var_power=var_power)

        dummy_exog = pd.DataFrame(np.ones((1, len(glm_cols_fitted_on))), columns=glm_cols_fitted_on)
        if 'const' not in dummy_exog.columns and 'const' in glm_cols_fitted_on:
             dummy_exog['const'] = 1.0
        dummy_endog = pd.Series([1.0])
        dummy_glm_model = sm.GLM(endog=dummy_endog, exog=dummy_exog[glm_cols_fitted_on], family=glm_family)
        
        pred_glm_series = dummy_glm_model.predict(params=glm_params_loaded, exog=X_glm_aligned.astype(float))
        pred_glm_val = pred_glm_series.iloc[0] if isinstance(pred_glm_series, pd.Series) else pred_glm_series[0]
        print("GLM prediction for single instance successful.")
    except Exception as e:
        print(f"Error making GLM prediction for single instance: {e}")
        pred_glm_val = np.nan

    # --- RF Prediction ---
    pred_rf_val = 0.0 # Default
    if rf_components_exist:
        try:
            rf_model_loaded = joblib.load(rf_model_file)
            rf_preprocessor_ct_loaded = joblib.load(rf_preprocessor_file)

            new_X_rf_processed, _, _ = create_rf_preprocessor(
                new_X_df.copy(), RF_DATE_FEATURES, RF_KNOWN_CATEGORICALS,
                RF_HIGH_CARDINALITY_THRESHOLD, RF_DROP_CARDINALITY_THRESHOLD,
                fit_preprocessor_flag=False,
                fitted_preprocessor_ct=rf_preprocessor_ct_loaded
            )
            if new_X_rf_processed.empty or new_X_rf_processed.shape[1] == 0:
                print("Warning: RF single instance data is empty after preprocessing. RF prediction will be zero.")
            else:
                pred_rf_array = rf_model_loaded.predict(new_X_rf_processed)
                pred_rf_val = pred_rf_array[0] if isinstance(pred_rf_array, np.ndarray) else pred_rf_array
                print("RF prediction for single instance successful.")
        except Exception as e:
            print(f"Error making RF prediction for single instance: {e}")
            pred_rf_val = np.nan
    else:
        print("RF components not found, RF prediction is 0 for single instance.")


    if np.isnan(pred_glm_val) or (rf_components_exist and np.isnan(pred_rf_val)): 
        print("Blending failed due to error in a base model prediction for single instance.")
        return np.nan
    
    blended_prediction_val = w_glm * pred_glm_val + w_rf * pred_rf_val
    print(f"Blending complete for single instance. GLM pred: {pred_glm_val:.4f}, RF pred: {pred_rf_val:.4f}, Blended: {blended_prediction_val:.4f}")
    return blended_prediction_val


def get_all_training_columns_except_target():
    """ Tries to get all feature column names from the original training dataset file. """
    try:
        if os.path.exists(DATASET_FILE):
            original_training_cols = pd.read_csv(DATASET_FILE, nrows=0).columns.tolist()
            # Remove Unnamed index if present
            if original_training_cols and original_training_cols[0].startswith('Unnamed: '):
                original_training_cols = original_training_cols[1:]
            # Remove target column
            if TARGET_COLUMN in original_training_cols:
                original_training_cols.remove(TARGET_COLUMN)
            # Remove columns dropped early
            for col_to_drop in COLUMNS_TO_DROP_EARLY:
                if col_to_drop in original_training_cols:
                    original_training_cols.remove(col_to_drop)
            return original_training_cols
    except Exception as e:
        print(f"Warning: Could not read original training columns from {DATASET_FILE}: {e}")
    return None


def single_predict_main():
    print("\n\n--- Example: Predicting on a single new data instance ---")
    
    single_raw_instance = {
        'SEX': 1, 'INSR_BEGIN': '15-AUG-22', 'INSR_END': '14-AUG-23', 'EFFECTIVE_YR': 2022,
        'INSR_TYPE': 1206, 'INSURED_VALUE': 100000.0, 'PREMIUM': 2500.75, 'PROD_YEAR': 2018,
        'SEATS_NUM': 5, 'CARRYING_CAPACITY': 0, 'TYPE_VEHICLE': '轿车', 'CCM_TON': 1998.0,
        'MAKE': '宝马', 'USAGE': '家用', 'CLAIM_PAID': 0.0, 'brand': '品牌A', 'average_speed': 55.0,
        'avg_daily_charges': 1, 'fatigue_driving_ratio': 0.05, 'late_night_trip_ratio': 0.1,
        'avg_late_night_trip_mileage': 10.5, 'high_temp_driving_ratio': 0.02, 'battery_type_lfp': 0,
        'initial_battery_soc': 85.0, 'avg_charge_duration': 7200.0,
        'insurance_commercial_third_party': 1, 'insurance_compulsory_third_party': 0
    }

    all_expected_features = get_all_training_columns_except_target()
    if all_expected_features:
        for col in all_expected_features:
            if col not in single_raw_instance:
                print(f"Note: Adding missing original training column '{col}' as np.nan to single instance.")
                single_raw_instance[col] = np.nan 
    else:
        print("Warning: Could not get full list of training columns. Single instance might be incomplete.")


    prediction = predict_single_instance_blended(single_raw_instance)
    if not np.isnan(prediction):
        print(f"\nPredicted '{TARGET_COLUMN}' for the single instance: {prediction:.4f}")
    else:
        print(f"\nPrediction for the single instance failed.")

def sample_new_predict_main():
    print(f"\n\n--- Example: Predicting on new data from {TEST_DATASET_FILE} ---") 
    
    try:
        sample_new_data = pd.read_csv(TEST_DATASET_FILE)
        print(f"Successfully loaded test data from {TEST_DATASET_FILE}. Shape: {sample_new_data.shape}")

        if sample_new_data.columns[0].startswith('Unnamed: '):
            sample_new_data = sample_new_data.drop(columns=[sample_new_data.columns[0]])
            print("Dropped 'Unnamed: ' column from test data.")

    except FileNotFoundError:
        print(f"Error: Test dataset file '{TEST_DATASET_FILE}' not found.")
        print("Please ensure the test data CSV is in the 'dataset' directory or update TEST_DATASET_FILE path.")
        print("Creating a dummy test dataset for demonstration as test file was not found...")
        num_new_samples = 10 
        start_date = datetime(2023, 1, 1)
        insr_begin_dates_new = [(start_date + timedelta(days=np.random.randint(0, 300))).strftime('%d-%b-%y').upper() for _ in range(num_new_samples)]
        insr_end_dates_new = [(pd.to_datetime(d, format='%d-%b-%y') + timedelta(days=365)).strftime('%d-%b-%y').upper() for d in insr_begin_dates_new]
        sample_new_data_dict = {
            'SEX': np.random.choice([1, 2], num_new_samples), 'INSR_BEGIN': insr_begin_dates_new,
            'INSR_END': insr_end_dates_new, 'EFFECTIVE_YR': np.random.choice([2023, 2024], num_new_samples),
            'INSR_TYPE': np.random.choice([1206, 1207, 101, 102], num_new_samples),
            'INSURED_VALUE': np.random.uniform(40000, 250000, num_new_samples),
            'PREMIUM': np.random.uniform(400, 6000, num_new_samples),
            'PROD_YEAR': np.random.randint(2012, 2024, num_new_samples),
            'SEATS_NUM': np.random.choice([2, 4, 5, 7, 8], num_new_samples),
            'CARRYING_CAPACITY': np.random.choice([0, 400, 800], num_new_samples),
            'TYPE_VEHICLE': np.random.choice(['轿车', 'SUV', 'MPV', '跑车', '新车型'], num_new_samples),
            'CCM_TON': np.random.uniform(1000, 6000, num_new_samples),
            'MAKE': np.random.choice(['宝马', '奔驰', '奥迪', '特斯拉', '比亚迪', '未知品牌'], num_new_samples),
            'USAGE': np.random.choice(['家用', '营运', '公务'], num_new_samples),
            'CLAIM_PAID': np.random.gamma(0.5, scale=3000, size=num_new_samples) * (np.random.rand(num_new_samples) < 0.1),
            'brand': np.random.choice(['品牌A', '品牌B', '品牌C', '新品牌E'], num_new_samples),
            'average_speed': np.random.uniform(10, 90, num_new_samples),
            'avg_daily_charges': np.random.poisson(1.5, num_new_samples),
            'fatigue_driving_ratio': np.random.uniform(0, 0.15, num_new_samples),
            'late_night_trip_ratio': np.random.uniform(0, 0.25, num_new_samples),
            'avg_late_night_trip_mileage': np.random.uniform(0, 60, num_new_samples) * (np.random.rand(num_new_samples) < 0.4),
            'high_temp_driving_ratio': np.random.uniform(0, 0.35, num_new_samples),
            'battery_type_lfp': np.random.choice([0, 1], num_new_samples, p=[0.6, 0.4]),
            'initial_battery_soc': np.random.uniform(15, 95, num_new_samples),
            'avg_charge_duration': np.random.uniform(2000, 40000, num_new_samples),
            'insurance_commercial_third_party': np.random.choice([0, 1], num_new_samples, p=[0.3, 0.7]),
            'insurance_compulsory_third_party': np.random.choice([0, 1], num_new_samples, p=[0.7, 0.3])
        }
        sample_new_data = pd.DataFrame(sample_new_data_dict)
        try:
            sample_new_data.to_csv(TEST_DATASET_FILE, index=False)
            print(f"Dummy test dataset saved to {TEST_DATASET_FILE}")
        except Exception as e_save_test:
            print(f"Could not save dummy test dataset: {e_save_test}")
    except Exception as e:
        print(f"An error occurred while loading or processing the test data from {TEST_DATASET_FILE}: {e}")
        return 

    all_expected_features = get_all_training_columns_except_target()
    if all_expected_features:
        for col in all_expected_features:
            if col not in sample_new_data.columns:
                print(f"Note: Adding missing original training column '{col}' as np.nan to test data.")
                sample_new_data[col] = np.nan 
    else:
        print("Warning: Could not get full list of training columns. Test data schema consistency check might be incomplete.")

    if TARGET_COLUMN in sample_new_data.columns:
        print(f"Note: Target column '{TARGET_COLUMN}' found in test data. It will be ignored for prediction.")
        
    if sample_new_data.empty:
        print("Test data is empty after loading and initial processing. Skipping prediction.")
        return

    final_blended_preds = predict_on_new_data(sample_new_data.copy()) 
    if final_blended_preds is not None:
        print("\nFinal Blended Predictions on New Sample Data (from file):")
        results_df = sample_new_data.copy()
        results_df['predicted_premium'] = final_blended_preds
        print(results_df[['PREMIUM', 'predicted_premium']].head())
        results_df.to_csv("blender_prediected.csv")
    else:
        print("Prediction on new sample data failed.")


if __name__ == '__main__':
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file {DATASET_FILE} not found. Generating a dummy one for the first run.")
        pass 

    main() 

    single_predict_main() 

    sample_new_predict_main()
