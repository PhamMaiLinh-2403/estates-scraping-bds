import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import optuna

# Suppress Optuna's informational messages
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Assuming your config file is in a 'src' directory
from src import config


def objective(trial, X, y):
    """
    The objective function for Optuna to optimize for the LightGBM model.
    """
    # --- 1. Define Hyperparameter Search Space ---
    params = {
        'objective': 'regression_l1',  # MAE, often more robust to outliers
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    # --- 2. Cross-Validation Setup ---
    N_SPLITS = 5
    y_binned = pd.cut(y, bins=10, labels=False, duplicates='drop')

    if y_binned.nunique() < N_SPLITS:
        y_stratify = y
    else:
        y_stratify = y_binned

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in cv.split(X, y_stratify):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold)

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        scores.append(rmse)

    return np.mean(scores)


def test_alley_width_model_accuracy():
    """
    Trains and compares multiple models for alley width prediction.
    - LightGBM: with Optuna for hyperparameter tuning.
    - Linear Regression: as a simple baseline.
    - Support Vector Regressor (SVR): as another powerful baseline.
    """
    print("--- Starting Model Accuracy Test and Comparison ---")

    # --- 1. Load Data ---
    if not os.path.exists(config.FEATURE_ENGINEERED_OUTPUT_FILE):
        print(f"ERROR: Feature engineered file not found at '{config.FEATURE_ENGINEERED_OUTPUT_FILE}'.")
        print("Please run the 'feature' pipeline step first (`--mode feature`).")
        return

    df_internal = pd.read_excel(config.FEATURE_ENGINEERED_OUTPUT_FILE)
    try:
        df_external = pd.read_excel(config.TRAIN_FILE)
    except FileNotFoundError:
        df_external = pd.DataFrame()
        print(f"WARNING: External training data not found at '{config.TRAIN_FILE}'.")
    df_full = pd.concat([df_internal, df_external], ignore_index=True)

    target_col = 'Độ rộng ngõ/ngách nhỏ nhất (m)'
    df_testable = df_full.dropna(subset=['Đơn giá đất', target_col]).copy()
    df_testable = df_testable[df_testable[target_col] != 0]

    if len(df_testable) < 50:
        print("Not enough testable data (< 50 records). Aborting test.")
        return
    print(f"Found {len(df_testable)} records with valid target values for testing.")

    # --- 2. Feature Preparation ---
    # numeric_features = [
    #     'Số tầng công trình', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)',
    #     'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp',
    #     'Khoảng cách tới trục đường chính (m)', 'Đơn giá đất'
    # ]
    # categorical_features = [
    #     'Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn',
    #     'Đường phố', 'Hình dạng'
    # ]

    numeric_features = [
        'Diện tích đất (m2)', 'Kích thước mặt tiền (m)',
        'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp',
        'Khoảng cách tới trục đường chính (m)', 'Đơn giá đất',
        'Số ngày tính từ lúc đăng tin'
    ]
    categorical_features = [
        'Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn',
        'Đường phố','Lợi thế kinh doanh', 'Hình dạng'
    ]
    cat_get_dummies = [
        'Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 
        'Xã/Phường/Thị trấn', 'Đường phố', 'Hình dạng'
    ]

    features = numeric_features + categorical_features

    X = df_testable[features]
    y = df_testable[target_col]

    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('Missing')

    # --- 3. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # --- 4. Preprocessing ---
    # One-Hot Encode categorical features
    def sanitize_column_name(col: str) -> str:
        return re.sub(r'[\[\]{},:"\\/]', '_', col)

    X_train_encoded = pd.get_dummies(X_train, columns=cat_get_dummies, dtype=float)
    X_train_encoded['Lợi thế kinh doanh'] = X_train['Lợi thế kinh doanh'].map({'Tốt': 4, 'Khá': 3, 'Trung bình': 2, 'Kém': 1, 'Missing': 0})
    X_test_encoded = pd.get_dummies(X_test, columns=cat_get_dummies, dtype=float)
    X_test_encoded['Lợi thế kinh doanh'] = X_test['Lợi thế kinh doanh'].map({'Tốt': 4, 'Khá': 3, 'Trung bình': 2, 'Kém': 1, 'Missing': 0})

    X_train_encoded.columns = [sanitize_column_name(c) for c in X_train_encoded.columns]
    X_test_encoded.columns = [sanitize_column_name(c) for c in X_test_encoded.columns]

    # Align columns
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_encoded[c] = 0
    extra_in_test = set(test_cols) - set(train_cols)
    X_test_encoded = X_test_encoded.drop(columns=list(extra_in_test))
    X_test_aligned = X_test_encoded[train_cols]

    # Scale features for Linear Regression and SVR (they are sensitive to feature scales)
    # Tree-based models like LightGBM do not require scaling.
    print("\nScaling features for Linear Regression and SVR...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_aligned)

    # --- 5. Hyperparameter Tuning for LightGBM ---
    print("\n--- Starting Hyperparameter Tuning for LightGBM with Optuna (using CV) ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_encoded, y_train), n_trials=100)

    print("\n--- Tuning Complete ---")
    print(f"Best CV RMSE for LightGBM: {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # --- 6. Train and Evaluate All Models on the Test Set ---
    # Define models
    lgbm_tuned = lgb.LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)
    linear_reg = LinearRegression()
    svr = SVR() # Using default parameters for SVR as a baseline

    models = {
        "Tuned LightGBM": lgbm_tuned,
        "Linear Regression": linear_reg,
        "SVR": svr
    }

    results = {}
    best_model_predictions = None

    for name, model in models.items():
        print(f"\n--- Training and Evaluating: {name} ---")

        # Use scaled data for LR and SVR, and original encoded data for LightGBM
        if name in ["Linear Regression", "SVR"]:
            X_train_fit = X_train_scaled
            X_test_predict = X_test_scaled
        else:
            X_train_fit = X_train_encoded
            X_test_predict = X_test_aligned

        # Train model
        model.fit(X_train_fit, y_train)

        # Make predictions
        predictions = model.predict(X_test_predict)

        # Evaluate
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        results[name] = {
            "R-squared": r2,
            "MAE (meters)": mae,
            "RMSE (meters)": rmse
        }
        print(f"Metrics for {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        # Save predictions of the tuned LGBM model for visualization
        if name == "Tuned LightGBM":
            best_model_predictions = predictions

    # --- 7. Compare Model Performance ---
    print("\n--- Final Model Performance Comparison (on Test Set) ---")
    results_df = pd.DataFrame(results).T
    # Sort by RMSE (lower is better)
    results_df = results_df.sort_values(by="RMSE (meters)", ascending=True)
    print(results_df.to_string(formatters={
        'R-squared': '{:.4f}'.format,
        'MAE (meters)': '{:.4f}'.format,
        'RMSE (meters)': '{:.4f}'.format
    }))
    best_model_name = results_df.index[0]
    print(f"\nBest performing model based on RMSE: {best_model_name}")

    # --- 8. Show a Sample of Predictions from the Best Model ---
    print(f"\n--- Sample Predictions vs. Actual Values (from {best_model_name}) ---")
    sample_results_df = pd.DataFrame({
        'Actual Width (m)': y_test,
        'Predicted Width (m)': best_model_predictions.round(2)
    })
    print(sample_results_df.head(10).to_string())

    # --- 9. Visualization for the Best Model ---
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_model_predictions, alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    plt.title(f'{best_model_name} Performance: Actual vs. Predicted on Test Set')
    plt.xlabel('Actual Alley Width (m)')
    plt.ylabel('Predicted Alley Width (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_alley_width_model_accuracy()