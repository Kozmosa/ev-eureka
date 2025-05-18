# blender.py
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

# --- Global Configuration ---
DATASET_DIR = 'dataset'
DATASET_FILE = os.path.join(DATASET_DIR, 'ev.csv')
TARGET_COLUMN = 'average_loss' # Updated Target Column

# For GLM
GLM_CATEGORICAL_FEATURES = ['brand'] # Updated

# For RF
RF_DATE_FEATURES = [] # Updated: No date features in the new format
RF_KNOWN_CATEGORICALS = ['brand'] # Updated: 'brand' is the main one
RF_HIGH_CARDINALITY_THRESHOLD = 100 # For OHE, e.g. if 'brand' had many levels
RF_DROP_CARDINALITY_THRESHOLD = 500 # For OHE, e.g. if 'brand' had many levels

N_SPLITS_CV = 5 # Number of folds for cross-validation
RANDOM_STATE = 42

# --- Model Paths (Optional, if you want to save/load) ---
MODELS_DIR = 'blender_models_v2' # Changed directory to avoid conflicts
GLM_MODEL_INFO_FILE = os.path.join(MODELS_DIR, 'glm_model_info.json')
RF_MODEL_COMPONENTS = {
    'model': os.path.join(MODELS_DIR, 'rf_full_model.joblib'),
    'preprocessor': os.path.join(MODELS_DIR, 'rf_full_preprocessor.joblib'),
    'feature_columns': os.path.join(MODELS_DIR, 'rf_full_feature_columns.json')
}
BLENDER_WEIGHTS_FILE = os.path.join(MODELS_DIR, 'blender_weights.json')

os.makedirs(MODELS_DIR, exist_ok=True)

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
            ct_transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_categorical_cols))

        if not ct_transformers:
            X_with_const = sm.add_constant(X_processed.astype(float), has_constant='add')
            return X_with_const, None

        preprocessor = ColumnTransformer(transformers=ct_transformers, remainder='passthrough')
        X_transformed = preprocessor.fit_transform(X_processed)
    elif preprocessor:
        X_transformed = preprocessor.transform(X_processed)
    else:
        X_with_const = sm.add_constant(X_processed.astype(float), has_constant='add')
        return X_with_const, None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError: # Older scikit-learn
        feature_names = []
        for name, trans, cols_list in preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out'): # For transformers like OneHotEncoder
                 feature_names.extend(trans.get_feature_names_out(cols_list))
            elif hasattr(trans, 'get_feature_names'): # older OHE
                 feature_names.extend(trans.get_feature_names(cols_list))
            elif name == 'num' or trans == 'passthrough': # For StandardScaler or passthrough
                feature_names.extend(cols_list)
            # Handle cases where 'remainder' columns are included if remainder='passthrough'
        if preprocessor.remainder == 'passthrough' and hasattr(preprocessor, 'feature_names_in_'):
            remainder_cols = [col for i, col in enumerate(preprocessor.feature_names_in_) if preprocessor._remainder[1][i]]
            feature_names.extend(remainder_cols)


    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_processed.index)
    X_transformed_df = sm.add_constant(X_transformed_df.astype(float), has_constant='add')
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

    try:
        glm_model_obj = sm.GLM(y_train_series, X_train_processed,
                               family=sm.families.Tweedie(link=sm.families.links.Log(), var_power=1.5))
        glm_results = glm_model_obj.fit()
        return glm_results, preprocessor, X_train_processed.columns.tolist()
    except Exception as e:
        print(f"Error training GLM: {e}")
        print("X_train_processed dtypes:\n", X_train_processed.dtypes)
        print("X_train_processed NaNs per column:\n", X_train_processed.isnull().sum()[X_train_processed.isnull().sum() > 0])
        print("y_train_series NaNs:", y_train_series.isnull().sum())
        print("y_train_series non-positive:", (y_train_series <=0).sum())
        raise

def predict_glm_on_fold(glm_results, X_val_df, preprocessor, feature_columns_glm):
    """Predicts using a trained GLM model on validation data."""
    X_val_processed, _ = preprocess_for_glm(
        X_val_df, [], [], fit_preprocessor=False, preprocessor=preprocessor
    )
    # Align columns to match training (reindex might be needed if preprocessor doesn't guarantee it)
    # If preprocessor was fitted on data with different columns, this alignment is critical.
    # However, preprocess_for_glm should output consistent columns based on preprocessor.get_feature_names_out()
    # For safety, ensure all expected columns are present, filling missing with 0 (after adding constant).
    # The constant is typically named 'const'.
    
    # Create a DataFrame with all expected feature_columns_glm, initialized to 0
    X_val_aligned = pd.DataFrame(0, index=X_val_processed.index, columns=feature_columns_glm)
    # Fill in the values from X_val_processed for columns that exist in both
    common_cols = X_val_aligned.columns.intersection(X_val_processed.columns)
    X_val_aligned[common_cols] = X_val_processed[common_cols]
    # Ensure 'const' column is 1 if it was added and is in feature_columns_glm
    if 'const' in feature_columns_glm and 'const' not in X_val_aligned: # Should be handled by add_constant
         X_val_aligned['const'] = 1.0 # Fallback, though add_constant should ensure it.
    elif 'const' in feature_columns_glm and 'const' in X_val_aligned:
         X_val_aligned['const'] = 1.0 # Ensure it is 1.0

    return glm_results.predict(X_val_aligned[feature_columns_glm])


# --- 2. Random Forest Model Training and Prediction Logic ---
def create_rf_preprocessor(X_df_input, # Renamed to avoid modifying original X_df
                           date_features, # Will be empty
                           known_categoricals,
                           high_card_thresh,
                           drop_card_thresh,
                           fit_preprocessor_flag=False): # Renamed to avoid conflict
    """Creates a preprocessor pipeline for RF."""
    X_transformed = X_df_input.copy()

    # 1. Date features (will be skipped if date_features is empty)
    for col in date_features:
        if col in X_transformed.columns:
            X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
            X_transformed[f'{col}_YEAR'] = X_transformed[col].dt.year
            X_transformed[f'{col}_MONTH'] = X_transformed[col].dt.month
            X_transformed[f'{col}_DAY'] = X_transformed[col].dt.day
    X_transformed = X_transformed.drop(columns=date_features, errors='ignore')

    # 2. Identify categorical and numerical features
    # Based on known_categoricals and dtypes
    categorical_features_identified = []
    numeric_features_identified = []

    for col in X_transformed.columns:
        if col in known_categoricals or X_transformed[col].dtype == 'object':
            if fit_preprocessor_flag: # Only check cardinality if fitting
                nunique = X_transformed[col].nunique()
                if nunique > drop_card_thresh:
                    print(f"RF Preprocessing: Dropping '{col}' due to very high cardinality ({nunique}).")
                    X_transformed.drop(columns=[col], inplace=True) # Drop from dataframe
                    continue # Skip adding to categorical_features_identified
                elif nunique > 0: # Ensure column is not all NaN
                    categorical_features_identified.append(col)
            elif col in X_transformed.columns: # If not fitting, just add if it exists (it will be transformed)
                 categorical_features_identified.append(col)

        elif pd.api.types.is_numeric_dtype(X_transformed[col]):
            numeric_features_identified.append(col)
        # else: column is of other type and not in known_categoricals, will be dropped by ColumnTransformer's remainder='drop'

    # Ensure no overlap and features exist
    categorical_features_identified = [c for c in categorical_features_identified if c in X_transformed.columns]
    numeric_features_identified = [n for n in numeric_features_identified if n in X_transformed.columns and n not in categorical_features_identified]


    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformers_list = []
    if numeric_features_identified:
        transformers_list.append(('num', numeric_transformer, numeric_features_identified))
    if categorical_features_identified:
        transformers_list.append(('cat', categorical_transformer, categorical_features_identified))

    if not transformers_list: # No features to transform
        return X_transformed, None, X_transformed.columns.tolist()

    preprocessor_ct = ColumnTransformer(transformers=transformers_list, remainder='drop')

    if fit_preprocessor_flag:
        X_final_array = preprocessor_ct.fit_transform(X_transformed)
        final_columns_out = preprocessor_ct.get_feature_names_out()
    else:
        # This branch should ideally not be called directly if preprocessor_ct is not fitted.
        # The intent is to pass a *fitted* preprocessor_ct for transformation.
        # If called with fit_preprocessor_flag=False, it means preprocessor_ct IS THE FITTED preprocessor.
        # So, we call transform on it.
        # The function signature is a bit confusing here. Let's assume if fit_preprocessor_flag=False,
        # then `preprocessor_ct` is actually the fitted preprocessor passed from outside.
        # This logic is handled by train_rf_on_fold and predict_rf_on_fold.
        print("Error: create_rf_preprocessor called with fit=False but this function creates a new one.")
        print("This path should not be taken in the current CV flow.")
        return None, None, None

    X_final_df = pd.DataFrame(X_final_array, columns=final_columns_out, index=X_df_input.index)
    return X_final_df, preprocessor_ct, final_columns_out


def train_rf_on_fold(X_train_df, y_train_series):
    """Trains a Random Forest model on a fold."""
    X_train_processed, preprocessor, feature_cols = create_rf_preprocessor(
        X_train_df,
        RF_DATE_FEATURES, RF_KNOWN_CATEGORICALS,
        RF_HIGH_CARDINALITY_THRESHOLD, RF_DROP_CARDINALITY_THRESHOLD,
        fit_preprocessor_flag=True # Fit the preprocessor
    )
    y_train_series = y_train_series.astype(float)

    rf_params = {
        'n_estimators': 100, 'max_features': 0.6, 'max_samples': 0.7,
        'max_depth': 9, 'min_samples_leaf': max(10, int(len(X_train_df) * 0.005)),
        'random_state': RANDOM_STATE, 'oob_score': False, 'n_jobs': -1
    }
    rf_model_obj = RandomForestRegressor(**rf_params)
    rf_model_obj.fit(X_train_processed, y_train_series)
    return rf_model_obj, preprocessor, feature_cols

def predict_rf_on_fold(rf_model, X_val_df, preprocessor):
    """Predicts using a trained RF model on validation data with its preprocessor."""
    # X_val_df is raw validation data. The preprocessor (fitted on train) will transform it.
    # The preprocessor should handle date features, known categoricals, etc., based on how it was fitted.
    # Our `create_rf_preprocessor` does date transforms *before* ColumnTransformer.
    # So, we need to replicate that for X_val_df before passing to the preprocessor's transform method.
    
    X_val_temp = X_val_df.copy()
    # Date features (will be skipped if RF_DATE_FEATURES is empty)
    for col in RF_DATE_FEATURES:
        if col in X_val_temp.columns:
            X_val_temp[col] = pd.to_datetime(X_val_temp[col], errors='coerce')
            X_val_temp[f'{col}_YEAR'] = X_val_temp[col].dt.year
            X_val_temp[f'{col}_MONTH'] = X_val_temp[col].dt.month
            X_val_temp[f'{col}_DAY'] = X_val_temp[col].dt.day
    X_val_temp = X_val_temp.drop(columns=RF_DATE_FEATURES, errors='ignore')

    # Now use the fitted preprocessor (ColumnTransformer part)
    X_val_transformed_array = preprocessor.transform(X_val_temp)
    return rf_model.predict(X_val_transformed_array)


# --- 3. Blending Weights Optimization ---
def loss_function(weights, pred_glm, pred_rf, y_true):
    """MSE loss for blending weights."""
    w_glm, w_rf = weights
    blended_pred = w_glm * pred_glm + w_rf * pred_rf
    return mean_squared_error(y_true, blended_pred)

def optimize_blending_weights(pred_glm_oof, pred_rf_oof, y_true_oof):
    """Finds optimal weights w_glm, w_rf."""
    constraints = ({'type': 'eq', 'fun': lambda weights: weights[0] + weights[1] - 1},
                   {'type': 'ineq', 'fun': lambda weights: weights[0]},
                   {'type': 'ineq', 'fun': lambda weights: weights[1]})
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
        # Drop the unnamed index column if it exists (from user's sample)
        if data_df.columns[0].startswith('Unnamed: '):
            data_df = data_df.drop(columns=[data_df.columns[0]])
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        print("Creating a dummy dataset for demonstration purposes based on new schema...")
        num_samples = 500
        data_df = pd.DataFrame({
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
            TARGET_COLUMN: np.random.gamma(2, scale=10, size=num_samples) + np.random.rand(num_samples) * 50 # More realistic loss
        })
        data_df[TARGET_COLUMN] = np.maximum(0.01, data_df[TARGET_COLUMN]) # Ensure positive loss

    if TARGET_COLUMN not in data_df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return

    data_df[TARGET_COLUMN] = data_df[TARGET_COLUMN].fillna(data_df[TARGET_COLUMN].median())
    y = data_df[TARGET_COLUMN]
    X = data_df.drop(columns=[TARGET_COLUMN])

    # Determine GLM numeric features
    glm_all_cols = X.columns.tolist()
    # Ensure GLM_CATEGORICAL_FEATURES are actually in the dataframe
    valid_glm_categorical_features = [col for col in GLM_CATEGORICAL_FEATURES if col in X.columns]
    glm_numeric_features = [
        col for col in glm_all_cols
        if col not in valid_glm_categorical_features and pd.api.types.is_numeric_dtype(X[col])
    ]
    print(f"GLM Categorical Features: {valid_glm_categorical_features}")
    print(f"GLM Numeric Features: {glm_numeric_features[:5]}... (Total: {len(glm_numeric_features)})")


    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
    oof_preds_glm = np.zeros(len(X))
    oof_preds_rf = np.zeros(len(X))
    oof_true_y = np.zeros(len(X))

    print(f"\nStarting {N_SPLITS_CV}-Fold Cross-Validation for OOF predictions...")
    for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"--- Fold {fold_idx + 1}/{N_SPLITS_CV} ---")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        print("Training GLM...")
        glm_model_fold, glm_preprocessor_fold, glm_cols_fold = train_glm_on_fold(
            X_train.copy(), y_train, valid_glm_categorical_features, glm_numeric_features
        )
        oof_preds_glm[val_index] = predict_glm_on_fold(glm_model_fold, X_val.copy(), glm_preprocessor_fold, glm_cols_fold)
        print("GLM OOF predictions for fold complete.")

        print("Training RF...")
        rf_model_fold, rf_preprocessor_fold, _ = train_rf_on_fold(X_train.copy(), y_train)
        oof_preds_rf[val_index] = predict_rf_on_fold(rf_model_fold, X_val.copy(), rf_preprocessor_fold)
        print("RF OOF predictions for fold complete.")

        oof_true_y[val_index] = y_val

    oof_preds_glm = np.nan_to_num(oof_preds_glm, nan=np.nanmean(oof_preds_glm[~np.isnan(oof_preds_glm)]))
    oof_preds_rf = np.nan_to_num(oof_preds_rf, nan=np.nanmean(oof_preds_rf[~np.isnan(oof_preds_rf)]))


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
    joblib.dump(final_glm_preprocessor, os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib'))
    joblib.dump(final_glm_model.params.to_dict(), os.path.join(MODELS_DIR, 'final_glm_model_params.json')) # Save params
    with open(GLM_MODEL_INFO_FILE, 'w') as f:
        json.dump({'columns_fitted': final_glm_cols,
                     'categorical_features_used': valid_glm_categorical_features,
                     'numeric_features_used': glm_numeric_features,
                     'family_info': {'class': 'Tweedie', 'link': 'Log', 'var_power': 1.5} # Store family info
                    }, f)
    print("Final GLM model (components) saved.")

    print("Training final RF model...")
    final_rf_model, final_rf_preprocessor, final_rf_cols = train_rf_on_fold(X.copy(), y)
    joblib.dump(final_rf_model, RF_MODEL_COMPONENTS['model'])
    joblib.dump(final_rf_preprocessor, RF_MODEL_COMPONENTS['preprocessor'])
    with open(RF_MODEL_COMPONENTS['feature_columns'], 'w') as f:
        json.dump(list(final_rf_cols), f)
    print("Final RF model and preprocessor saved.")
    print("\n--- Blending Process Complete ---")

def predict_on_new_data(new_X_df_input):
    print("\n--- Predicting on New Data (Example) ---")
    if not all(os.path.exists(f) for f in [
        RF_MODEL_COMPONENTS['model'], RF_MODEL_COMPONENTS['preprocessor'],
        RF_MODEL_COMPONENTS['feature_columns'], BLENDER_WEIGHTS_FILE,
        os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib'),
        os.path.join(MODELS_DIR, 'final_glm_model_params.json'),
        GLM_MODEL_INFO_FILE
    ]):
        print("Error: Not all model components found. Run main training first.")
        return None

    with open(BLENDER_WEIGHTS_FILE, 'r') as f:
        blender_info = json.load(f)
    w_glm, w_rf = blender_info['w_glm'], blender_info['w_rf']
    print(f"Loaded blender weights: GLM={w_glm:.4f}, RF={w_rf:.4f}")

    new_X_df = new_X_df_input.copy()

    # --- GLM Prediction on New Data ---
    print("Preparing GLM prediction...")
    glm_preprocessor_loaded = joblib.load(os.path.join(MODELS_DIR, 'final_glm_preprocessor.joblib'))
    glm_params_loaded = pd.Series(joblib.load(os.path.join(MODELS_DIR, 'final_glm_model_params.json')))
    with open(GLM_MODEL_INFO_FILE, 'r') as f:
        glm_info = json.load(f)
    glm_cols_fitted_on = glm_info['columns_fitted']
    
    new_X_glm_processed, _ = preprocess_for_glm(
        new_X_df, [], [], fit_preprocessor=False, preprocessor=glm_preprocessor_loaded
    )
    # Align columns post-transformation to match exactly what GLM was fitted on
    X_glm_aligned = pd.DataFrame(0, index=new_X_glm_processed.index, columns=glm_cols_fitted_on)
    common_cols = X_glm_aligned.columns.intersection(new_X_glm_processed.columns)
    X_glm_aligned[common_cols] = new_X_glm_processed[common_cols]
    if 'const' in glm_cols_fitted_on: X_glm_aligned['const'] = 1.0


    # Reconstruct a minimal GLM model object for prediction
    # This avoids refitting on full original data for prediction, using saved parameters.
    family_info = glm_info.get('family_info', {'class': 'Tweedie', 'link': 'Log', 'var_power': 1.5})
    link_func = sm.families.links.Log() # Assuming Log link
    if family_info['link'].lower() == 'identity': link_func = sm.families.links.identity()
    # Add other links if necessary
    
    glm_family = sm.families.Tweedie(link=link_func, var_power=family_info['var_power'])
    
    # Create an unfitted model instance
    exog = pd.DataFrame(np.zeros((1, len(glm_cols_fitted_on))), columns=glm_cols_fitted_on) # Dummy exog for init
    endog = pd.Series([1]) # Dummy endog for init
    dummy_glm_model = sm.GLM(endog=endog, exog=exog, family=glm_family)
    
    # Predict using the predict method of the model class itself, with loaded params
    try:
        pred_glm_new = dummy_glm_model.predict(params=glm_params_loaded, exog=X_glm_aligned[glm_cols_fitted_on])
        print("GLM prediction on new data successful (using loaded params).")
    except Exception as e:
        print(f"Error making GLM prediction with loaded params: {e}")
        print("GLM predictions will be zero for blending.")
        pred_glm_new = np.zeros(len(new_X_df))


    # --- RF Prediction on New Data ---
    print("Preparing RF prediction...")
    rf_model_loaded = joblib.load(RF_MODEL_COMPONENTS['model'])
    rf_preprocessor_loaded = joblib.load(RF_MODEL_COMPONENTS['preprocessor'])
    # rf_feature_columns_loaded = json.load(open(RF_MODEL_COMPONENTS['feature_columns'], 'r')) # Not directly used if preprocessor applied

    X_rf_temp = new_X_df.copy()
    for col in RF_DATE_FEATURES: # Will be skipped if empty
        if col in X_rf_temp.columns:
            X_rf_temp[col] = pd.to_datetime(X_rf_temp[col], errors='coerce')
            # ... (add year, month, day extraction if needed) ...
    X_rf_temp = X_rf_temp.drop(columns=RF_DATE_FEATURES, errors='ignore')
    
    new_X_rf_processed_array = rf_preprocessor_loaded.transform(X_rf_temp)
    pred_rf_new = rf_model_loaded.predict(new_X_rf_processed_array)
    print("RF prediction on new data successful.")

    blended_predictions_new = w_glm * pred_glm_new + w_rf * pred_rf_new
    print("Blending complete for new data.")
    return blended_predictions_new


if __name__ == '__main__':
    main()

    print("\n\n--- Example: Predicting on sample new data ---")
    num_new_samples = 5
    sample_new_data = pd.DataFrame({
        'brand': np.random.choice(['品牌A', '品牌B', '品牌C', '品牌D', '新品牌E'], num_new_samples), # Added new brand
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
    })
    # Ensure all original columns from training data (if known) are present in sample_new_data,
    # even if as NaN, for preprocessors to handle them consistently.
    try:
        # Attempt to load column list from the original dataset if it exists
        # This helps ensure the sample_new_data has a schema consistent with training data
        if os.path.exists(DATASET_FILE):
            original_cols = pd.read_csv(DATASET_FILE, nrows=0).columns.tolist()
            if original_cols[0].startswith('Unnamed: '): # Drop index col from list
                original_cols = original_cols[1:]
            if TARGET_COLUMN in original_cols:
                original_cols.remove(TARGET_COLUMN)

            for col in original_cols:
                if col not in sample_new_data.columns:
                    sample_new_data[col] = np.nan # Add missing columns as NaN for imputation
    except Exception as e:
        print(f"Note: Could not align sample_new_data columns with original dataset: {e}")


    final_blended_preds = predict_on_new_data(sample_new_data)
    if final_blended_preds is not None:
        print("\nFinal Blended Predictions on New Sample Data:")
        print(pd.Series(final_blended_preds, index=sample_new_data.index).head())