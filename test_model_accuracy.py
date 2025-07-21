import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import optuna

# Suppress Optuna's informational messages
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Assuming your config file is in a 'src' directory
from src import config


def objective(trial, X, y):
    """
    The objective function for Optuna to optimize.
    It trains a model with a given set of hyperparameters and evaluates it using cross-validation.
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
    # StratifiedKFold is better for imbalanced datasets. For regression, we can
    # bin the target variable to create "strata" to ensure each fold has a
    # similar distribution of target values.
    N_SPLITS = 5
    # Create bins for stratification. Using 10 bins is a reasonable starting point.
    # `duplicates='drop'` handles cases where bin edges are not unique.
    y_binned = pd.cut(y, bins=10, labels=False, duplicates='drop')

    # If binning results in fewer than N_SPLITS unique bins, fallback to using original y
    if y_binned.nunique() < N_SPLITS:
        y_stratify = y
    else:
        y_stratify = y_binned

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    scores = []
    # Use the binned y for splitting, but the original y for training/evaluation
    for train_idx, val_idx in cv.split(X, y_stratify):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_fold, y_train_fold)

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        scores.append(rmse)

    # Optuna aims to minimize this returned value
    return np.mean(scores)


def test_alley_width_model_accuracy():
    """
    Trains and evaluates the alley width prediction model using Optuna for hyperparameter
    tuning and a final train-test split for validation.
    """
    print("--- Starting Model Accuracy Test with Hyperparameter Tuning ---")

    # --- 1. Load Data (Same as before) ---
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

    # --- 2. Feature Preparation (Same as before) ---
    numeric_features = [
        'Số tầng công trình', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)',
        'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp',
        'Khoảng cách tới trục đường chính (m)', 'Đơn giá đất'
    ]
    categorical_features = [
        'Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn',
        'Đường phố', 'Hình dạng'
    ]
    features = numeric_features + categorical_features

    X = df_testable[features]
    y = df_testable[target_col]

    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('Missing')

    # --- 3. Train-Test Split (Same as before) ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # --- 4. Preprocessing (Same as before) ---
    def sanitize_column_name(col: str) -> str:
        return re.sub(r'[\[\]{},:"\\/]', '_', col)

    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, dtype=float)

    X_train_encoded.columns = [sanitize_column_name(c) for c in X_train_encoded.columns]
    X_test_encoded.columns = [sanitize_column_name(c) for c in X_test_encoded.columns]

    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_encoded[c] = 0
    extra_in_test = set(test_cols) - set(train_cols)
    X_test_encoded = X_test_encoded.drop(columns=list(extra_in_test))
    X_test_aligned = X_test_encoded[train_cols]

    # --- 5. Hyperparameter Tuning with Optuna & CV ---
    print("\n--- Starting Hyperparameter Tuning with Optuna (using Cross-Validation) ---")
    study = optuna.create_study(direction='minimize')
    # We pass the training data to the objective function using a lambda
    study.optimize(lambda trial: objective(trial, X_train_encoded, y_train), n_trials=100)

    print("\n--- Tuning Complete ---")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    print("Best parameters found:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")

    # --- 6. Train Final Model with Best Parameters ---
    print("\nTraining the final model on the full training set with best parameters...")
    best_params = study.best_params
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbosity=-1)
    final_model.fit(X_train_encoded, y_train)
    print("Final model training complete.")

    # --- 7. Prediction & Evaluation on the Held-Out Test Set ---
    print("\nMaking predictions on the unseen test set...")
    predictions = final_model.predict(X_test_aligned)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Final Tuned Model Performance Metrics (on Test Set) ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"  - Interpretation: The model explains {r2:.1%} of the variance in the alley width.")
    print(f"Mean Absolute Error (MAE): {mae:.4f} meters")
    print(f"  - Interpretation: On average, the model's prediction is off by {mae:.2f} meters.")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} meters")
    print(f"  - Interpretation: A measure similar to MAE, but penalizes larger errors more heavily.")

    # --- 8. Show a Sample of Predictions ---
    print("\n--- Sample Predictions vs. Actual Values ---")
    results_df = pd.DataFrame({
        'Actual Width (m)': y_test,
        'Predicted Width (m)': predictions.round(2)
    })
    print(results_df.head(10).to_string())

    # --- 9. Visualization ---
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    plt.title('Tuned Model Performance: Actual vs. Predicted on Test Set')
    plt.xlabel('Actual Alley Width (m)')
    plt.ylabel('Predicted Alley Width (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_alley_width_model_accuracy()