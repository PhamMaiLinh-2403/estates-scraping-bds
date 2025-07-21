import os
import re
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Assuming your config file is in a 'src' directory
from src import config


def test_alley_width_model_accuracy():
    """
    Trains and evaluates the alley width prediction model using a train-test split.
    This function mimics the data preparation pipeline from main.py to ensure a valid test.
    """
    print("--- Starting Model Accuracy Test ---")

    # --- 1. Load Data ---
    # This section mirrors the data loading logic in `_predict_alley_width_ml_step`

    # Check for required files
    if not os.path.exists(config.FEATURE_ENGINEERED_OUTPUT_FILE):
        print(f"ERROR: Feature engineered file not found at '{config.FEATURE_ENGINEERED_OUTPUT_FILE}'.")
        print("Please run the 'feature' pipeline step first (`--mode feature`).")
        return

    # Load the data that was processed by the pipeline
    df_internal = pd.read_excel(config.FEATURE_ENGINEERED_OUTPUT_FILE)
    print(f"Loaded {len(df_internal)} records from the pipeline output.")

    # Load external training data
    try:
        df_external = pd.read_excel(config.TRAIN_FILE)
        print(f"Loaded {len(df_external)} external records from '{config.TRAIN_FILE}'.")
    except FileNotFoundError:
        df_external = pd.DataFrame()
        print(f"WARNING: External training data not found at '{config.TRAIN_FILE}'. Proceeding without it.")

    # Combine datasets
    df_full = pd.concat([df_internal, df_external], ignore_index=True)

    # We can only test on data where we have a valid, known target value.
    target_col = 'Độ rộng ngõ/ngách nhỏ nhất (m)'
    df_testable = df_full.dropna(subset=['Đơn giá đất', target_col]).copy()
    df_testable = df_testable[df_testable[target_col] != 0]

    if len(df_testable) < 50:
        print("Not enough testable data (< 50 records with valid alley widths). Aborting test.")
        return

    print(f"Found {len(df_testable)} records with valid target values for testing.")

    # --- 2. Feature Preparation ---
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

    # Impute missing values consistently with the main pipeline
    for col in numeric_features:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        X[col] = X[col].astype(str).fillna('Missing')

    # --- 3. Train-Test Split ---
    # Split the data BEFORE one-hot encoding to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # --- 4. Preprocessing (One-Hot Encoding & Sanitization) ---
    def sanitize_column_name(col: str) -> str:
        """Removes special characters from column names for LightGBM compatibility."""
        return re.sub(r'[\[\]{},:"\\/]', '_', col)

    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, dtype=float)

    # Sanitize column names
    X_train_encoded.columns = [sanitize_column_name(c) for c in X_train_encoded.columns]
    X_test_encoded.columns = [sanitize_column_name(c) for c in X_test_encoded.columns]

    # Align columns: ensures test set has the same columns as the training set
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns

    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_encoded[c] = 0

    extra_in_test = set(test_cols) - set(train_cols)
    X_test_encoded = X_test_encoded.drop(columns=list(extra_in_test))

    X_test_aligned = X_test_encoded[train_cols]

    # --- 5. Model Training ---
    print("\nTraining the LightGBM model...")
    model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    model.fit(X_train_encoded, y_train)
    print("Model training complete.")

    # --- 6. Prediction & Evaluation ---
    print("Making predictions on the test set...")
    predictions = model.predict(X_test_aligned)

    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n--- Model Performance Metrics ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"  - Interpretation: The model explains {r2:.1%} of the variance in the alley width.")
    print(f"Mean Absolute Error (MAE): {mae:.4f} meters")
    print(f"  - Interpretation: On average, the model's prediction is off by {mae:.2f} meters.")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} meters")
    print(f"  - Interpretation: A measure similar to MAE, but penalizes larger errors more heavily.")

    # --- 7. Show a Sample of Predictions ---
    print("\n--- Sample Predictions vs. Actual Values ---")
    results_df = pd.DataFrame({
        'Actual Width (m)': y_test,
        'Predicted Width (m)': predictions.round(2)
    })
    print(results_df.head(10).to_string())

    # --- 8. Visualization ---
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5, label='Predicted vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    plt.title('Alley Width Prediction: Actual vs. Predicted')
    plt.xlabel('Actual Alley Width (m)')
    plt.ylabel('Predicted Alley Width (m)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test_alley_width_model_accuracy()