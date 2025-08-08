import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from . import config


def predict_alley_width(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict and fill missing 'Độ rộng ngõ/ngách nhỏ nhất (m)' values using LightGBM.
    Applies log transform on the target before train-test split and evaluates accuracy.
    """
    target_col = 'Độ rộng ngõ/ngách nhỏ nhất (m)'
    df_copy = df.copy()

    df_internal_train = df_copy[df_copy['Đơn giá đất'].notna() & df_copy[target_col].notna() & (df_copy[target_col] != 0)]
    print(f"- Found {len(df_internal_train)} valid internal records for ML training.")

    try:
        df_external_train = pd.read_excel(config.TRAIN_FILE)
        df_external_train.columns = ['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn',
       'Đường phố', 'Chi tiết', 'Nguồn thông tin',
       'Tình trạng giao dịch', 'Thời điểm giao dịch/rao bán',
       'Thông tin liên hệ', 'Giá rao bán/giao dịch', 'Giá ước tính',
       'Loại đơn giá (đ/m2 hoặc đ/m ngang)', 'Đơn giá đất',
       'Số tầng công trình', 'Chất lượng còn lại',
       'Giá trị công trình xây dựng', 'Diện tích đất (m2)',
       'Tổng diện tích sàn', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)',
       'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)',
       'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất',
       'Hình ảnh của bài đăng', 'Ảnh chụp màn hình thông tin thu thập',
       'Yếu tố khác', 'Lợi thế kinh doanh']
        print(f"- Loaded {len(df_external_train)} external records from '{config.TRAIN_FILE}'.")
    except FileNotFoundError:
        df_external_train = pd.DataFrame()
        print(f"- WARNING: External training data not found at '{config.TRAIN_FILE}'.")

    print(f'Calculating additional features for One Housing...')

    df_external_train['Số ngày tính từ lúc đăng tin'] = 0

    numeric_features = [
        'Diện tích đất (m2)', 'Kích thước mặt tiền (m)',
        'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp',
        'Khoảng cách tới trục đường chính (m)', 'Đơn giá đất',
        'Số ngày tính từ lúc đăng tin'
    ]
    categorical_features = [
        'Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn',
        'Đường phố', 'Lợi thế kinh doanh', 'Hình dạng'
    ]
    features = numeric_features + categorical_features

    df_train = pd.concat([df_external_train, df_internal_train], ignore_index=True)
    initial_train_count = len(df_train)
    df_train.dropna(subset=['Đơn giá đất', target_col], inplace=True)
    df_train = df_train[df_train[target_col] != 0].copy()

    print(f"- Combined internal and external data, resulting in {len(df_train)} clean training records (dropped {initial_train_count - len(df_train)} rows).")

    if len(df_train) < 50:
        print("- Not enough training data (< 50 records). Skipping alley width prediction.")
        return df

    X = df_train[features].copy()
    y = np.log1p(df_train[target_col])

    for col in numeric_features:
        print(f'NaN values in "{col}": {X[col].isna().sum()}')
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_features:
        old_length = X.shape[0]
        na_index = X[X[col].isna()].index
        X.dropna(subset=[col], inplace=True)
        y.drop(na_index, inplace=True)
        new_length = X.shape[0]
        print(f"- Dropped {old_length - new_length} records with missing '{col}'.")

    # Manual ordinal encoding for 'Lợi thế kinh doanh'
    X['Lợi thế kinh doanh'] = X['Lợi thế kinh doanh'].map({'Tốt': 4, 'Khá': 3, 'Trung bình': 2, 'Kém': 1, 'Missing': 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # Identify categorical features for LightGBM
    lgb_categorical = [col for col in categorical_features if col != 'Lợi thế kinh doanh']
    X_train = pd.get_dummies(X_train, columns=lgb_categorical)
    X_test = pd.get_dummies(X_test, columns=lgb_categorical)

    # Remove banned patterns for column names
    def sanitize_column_name(col: str) -> str:
        return re.sub(r'[\[\]{},:"\\/]', '_', col)
    X_train.columns = [sanitize_column_name(col) for col in X_train.columns]
    X_test.columns = [sanitize_column_name(col) for col in X_test.columns]

    # Align columns
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    extra_in_test = set(test_cols) - set(train_cols)
    X_test = X_test.drop(columns=list(extra_in_test))
    X_test_aligned = X_test[train_cols]

    # Train LightGBM
    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    log_preds = model.predict(X_test_aligned)
    preds = np.expm1(log_preds)
    y_test_true = np.expm1(y_test)

    # Predict missing
    predict_mask = df[target_col].isna()
    if not predict_mask.any():
        print("- No missing alley widths to predict.")
        return df

    df_to_predict = df[predict_mask]
    X_predict = df_to_predict[features].copy()
    for col in numeric_features:
        X_predict[col] = X_predict[col].fillna(X[col].median())
    for col in categorical_features:
        X_predict[col] = X_predict[col].astype(str).fillna('Missing')
    X_predict['Lợi thế kinh doanh'] = X_predict['Lợi thế kinh doanh'].map({'Tốt': 4, 'Khá': 3, 'Trung bình': 2, 'Kém': 1, 'Missing': 0})
    X_predict = pd.get_dummies(X_predict, columns=lgb_categorical)
    X_predict.columns = [sanitize_column_name(col) for col in X_predict.columns]

    # Align columns
    train_cols = X_train.columns
    test_cols = X_predict.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_predict[c] = 0
    extra_in_test = set(test_cols) - set(train_cols)
    X_predict = X_predict.drop(columns=list(extra_in_test))
    X_predict = X_predict[train_cols]

    log_predictions = model.predict(X_predict)
    predictions = np.expm1(log_predictions)
    predictions[predictions < 0] = 0

    df.loc[predict_mask, target_col] = [round(p, 2) for p in predictions]
    print(f'- Mean absolute error: {mean_absolute_error(y_test_true, preds):.3f}')
    print(f"- RMSE: {root_mean_squared_error(y_test_true, preds):.3f}")
    print(f'- R2 score: {r2_score(y_test_true, preds):.3f}')
    print(f"- Successfully filled {len(predictions)} missing values for '{target_col}'.")
    return df