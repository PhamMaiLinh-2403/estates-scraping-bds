import argparse
import csv
import os
import random
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from src.selenium_manager import create_stealth_driver
from src.scraping import Scraper
from src.cleaning import DataCleaner, drop_mixed_listings
from src.feature_engineering import FeatureEngineer
from src.address_standardizer import AddressStandardizer
from src import config


# --- File I/O ---
def save_urls_to_csv(urls, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        writer.writerows([[u] for u in urls])

    print(f"Saved {len(urls)} URLs → {file_path}")


def save_details_to_csv(details, file_path):
    """
    Saves scraped details to a CSV file, supporting both overwrite and append modes.
    The behavior is controlled by the `append_mode` setting in `config.py`.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not details:
        print("No new details to save.")
        return

    # Check if the file exists to determine if we need to write the header
    file_exists = os.path.exists(file_path)

    # Read the append_mode from config; default to False if not found
    is_append_mode = config.SCRAPING_DETAILS_CONFIG.get("append_mode", False)

    # Determine mode and header based on config and file existence
    mode = 'a' if is_append_mode and file_exists else 'w'
    write_header = not (is_append_mode and file_exists)

    df = pd.DataFrame(details)
    df.to_csv(
        file_path,
        mode=mode,
        header=write_header,
        index=False,
        quoting=csv.QUOTE_ALL
    )

    if mode == 'a':
        print(f"Appended {len(details)} new listing records → {file_path}")
    else:
        print(f"Saved {len(details)} listing records → {file_path}")

# --- Worker & Concurrency ---
def chunks(iterable, n):
    iterable = list(iterable)
    k, m = divmod(len(iterable), n)
    start = 0

    for i in range(n):
        end = start + k + (1 if i < m else 0)
        yield iterable[start:end]
        start = end


def scrape_worker(worker_id: int, url_subset: list[str]) -> list[dict]:
    """Each worker gets its own driver & scraper."""
    base = config.SCRAPING_DETAILS_CONFIG.get("stagger_step_sec", 2.0)
    start_delay = worker_id * base
    print(f"[Worker {worker_id}]: Sleeping {start_delay:.1f}s before start.")
    time.sleep(start_delay)

    driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
    scraper = Scraper(driver)
    results = []

    for idx, url in enumerate(url_subset, 1):
        print(f"[Worker {worker_id}]  {idx}/{len(url_subset)}  → {url}")
        data = scraper.scrape_listing_details(url)

        if data:
            results.append(data)
        if config.SCRAPING_DETAILS_CONFIG["stagger_mode"] == "random":
            delay = random.uniform(
                config.SCRAPING_DETAILS_CONFIG["stagger_step_sec"],
                config.SCRAPING_DETAILS_CONFIG["stagger_max_sec"],
            )
            time.sleep(delay)
    driver.quit()
    return results

def _predict_alley_width_ml_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper to predict and fill 'Độ rộng ngõ/ngách nhỏ nhất (m)' using a Support Vector Regressor (SVR) model.
    This version uses a log-transform on the target and scales the features.
    """
    target_col = 'Độ rộng ngõ/ngách nhỏ nhất (m)'
    df_copy = df.copy()

    # --- 1. Prepare Training Data ---
    df_internal_train = df_copy[df_copy['Đơn giá đất'].notna() & df_copy[target_col].notna() & (df_copy[target_col] != 0)]
    print(f"- Found {len(df_internal_train)} valid internal records for ML training.")

    try:
        df_external_train = pd.read_excel(config.TRAIN_FILE)
        print(f"- Loaded {len(df_external_train)} external records from '{config.TRAIN_FILE}'.")
    except FileNotFoundError:
        df_external_train = pd.DataFrame()
        print(f"- WARNING: External training data not found at '{config.TRAIN_FILE}'.")

    df_train = pd.concat([df_external_train, df_internal_train], ignore_index=True)

    # --- FIX: Apply cleaning steps AFTER concatenation to ensure both internal and external data are clean ---
    # 1. Drop rows where essential columns for training (unit price, target) are missing.
    initial_train_count = len(df_train)
    df_train.dropna(subset=['Đơn giá đất', target_col], inplace=True)

    # 2. Filter out rows where the target is 0, as this is invalid data for this prediction.
    df_train = df_train[df_train[target_col] != 0].copy()

    print(f"- Combined internal and external data, resulting in {len(df_train)} clean training records (dropped {initial_train_count - len(df_train)} rows with missing values).")

    if len(df_train) < 50:
        print("- Not enough training data (< 50 records). Skipping alley width prediction.")
        return df

    # --- 2. Feature Engineering & Preprocessing for ML ---
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

    X_train = df_train[features].copy()
    y_train = df_train[target_col].copy()

    # LOG TRANSFORM THE TARGET VARIABLE
    y_train_log = np.log1p(y_train)

    for col in numeric_features:
        X_train[col] = X_train[col].fillna(X_train[col].median())
    for col in categorical_features:
        X_train[col] = X_train[col].astype(str).fillna('Missing')

    def sanitize_column_name(col: str) -> str:
        return re.sub(r'[\[\]{},:"\\/]', '_', col)

    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, dtype=float)
    X_train_encoded.columns = [sanitize_column_name(c) for c in X_train_encoded.columns]

    # SCALE FEATURES (CRITICAL FOR SVR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)

    # --- 3. Model Training ---
    print(f"  - Training SVR model on {len(X_train_scaled)} records...")
    # Using SVR model instead of LightGBM
    model = SVR()
    model.fit(X_train_scaled, y_train_log)

    # --- 4. Prediction ---
    predict_mask = df[target_col].isna()
    if not predict_mask.any():
        print("- No missing alley widths to predict.")
        return df

    df_to_predict = df[predict_mask]
    X_predict = df_to_predict[features].copy()

    for col in numeric_features:
        X_predict[col] = X_predict[col].fillna(X_train[col].median())
    for col in categorical_features:
        X_predict[col] = X_predict[col].astype(str).fillna('Missing')

    X_predict_encoded = pd.get_dummies(X_predict, columns=categorical_features, dtype=float)
    X_predict_encoded.columns = [sanitize_column_name(c) for c in X_predict_encoded.columns]

    train_cols = X_train_encoded.columns
    predict_cols = X_predict_encoded.columns
    missing_in_predict = set(train_cols) - set(predict_cols)
    for c in missing_in_predict:
        X_predict_encoded[c] = 0
    extra_in_predict = set(predict_cols) - set(train_cols)
    X_predict_encoded.drop(columns=list(extra_in_predict), inplace=True)
    X_predict_aligned = X_predict_encoded[train_cols]

    # SCALE PREDICTION DATA USING THE SAME SCALER
    X_predict_scaled = scaler.transform(X_predict_aligned)

    print(f"  - Predicting {len(X_predict_scaled)} missing alley width values...")
    log_predictions = model.predict(X_predict_scaled)

    # INVERSE TRANSFORM PREDICTIONS
    predictions = np.expm1(log_predictions)
    # Ensure predictions are not negative
    predictions[predictions < 0] = 0

    # Update the original DataFrame
    df.loc[predict_mask, target_col] = [round(p, 2) for p in predictions]
    print(f"  - Successfully filled {len(predictions)} missing values for '{target_col}'.")
    return df


# --- Pipeline Steps ---
def run_scrape_urls():
    """Step 1: Scrape listing URLs from search pages."""
    driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
    scraper = Scraper(driver)
    urls = scraper.scrape_listing_urls(config.SEARCH_PAGE_URL)
    save_urls_to_csv(urls, config.URLS_OUTPUT_FILE)
    driver.quit()


def run_scrape_details():
    """Step 2: Scrape detailed information for each URL."""
    if not os.path.exists(config.URLS_OUTPUT_FILE):
        print("URL file not found. Run with `--mode urls` first.")
        return

    urls = pd.read_csv(config.URLS_OUTPUT_FILE)["url"].tolist()
    print(f"Loaded {len(urls)} URLs for detail‑scrape.")

    start = config.SCRAPING_DETAILS_CONFIG["start_index"]
    count = config.SCRAPING_DETAILS_CONFIG["count"]
    urls_to_scrape = urls[start: start + count] if count else urls[start:]

    max_workers = min(config.MAX_WORKERS, len(urls_to_scrape))
    if max_workers == 0:
        print("No URLs to scrape.")
        return

    url_chunks = list(chunks(urls_to_scrape, max_workers))
    print(f"Spawning {max_workers} workers (≈{[len(c) for c in url_chunks]} URLs per worker).")
    details_all = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(scrape_worker, wid, subset): wid
            for wid, subset in enumerate(url_chunks)
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            try:
                worker_details = fut.result()
                if worker_details:
                    details_all.extend(worker_details)
                print(f"[Worker {wid}]  ✔  {len(worker_details or [])} listings scraped.")
            except Exception as exc:
                print(f"[Worker {wid}]  ❌  raised {exc!r}")
    save_details_to_csv(details_all, config.DETAILS_OUTPUT_FILE)


def run_cleaning_pipeline():
    """Step 3: Clean the raw data and structure it."""
    if not os.path.exists(config.DETAILS_OUTPUT_FILE):
        print(f"Raw details file not found: {config.DETAILS_OUTPUT_FILE}. Run with `--mode details` first.")
        return

    print(f"Reading raw data from '{config.DETAILS_OUTPUT_FILE}'...")
    df_raw = pd.read_csv(config.DETAILS_OUTPUT_FILE)
    df_raw = drop_mixed_listings(df_raw)

    cleaned_records = []
    for _, row in df_raw.iterrows():
        row_dict = row.to_dict()
        direct_features = DataCleaner.extract_direct_features(row_dict)

        processed_data = {
            'Tỉnh/Thành phố': DataCleaner.extract_city(row_dict),
            'Thành phố/Quận/Huyện/Thị xã': DataCleaner.extract_district(row_dict),
            'Xã/Phường/Thị trấn': DataCleaner.extract_ward(row_dict),
            'Đường phố': DataCleaner.extract_street(row_dict),
            'Chi tiết': DataCleaner.extract_address_detail(row_dict),
            'Nguồn thông tin': row_dict.get('url'),
            'Tình trạng giao dịch': 'Rao bán',
            'Thời điểm giao dịch/rao bán': DataCleaner.extract_published_date(row_dict.get('main_info')),
            'Thông tin liên hệ': None,
            'Giá rao bán/giao dịch': DataCleaner.extract_total_price(row_dict.get('main_info')),
            'Loại đơn giá (đ/m2 hoặc đ/m ngang)': 'đ/m2',
            'Số tầng công trình': DataCleaner.extract_num_floors(row_dict),
            'Tổng diện tích sàn': DataCleaner.extract_built_area(row_dict),
            'Đơn giá xây dựng': DataCleaner.get_construction_cost(row_dict),
            'Năm xây dựng': None,
            'Chất lượng còn lại': DataCleaner.estimate_remaining_quality(row_dict),
            'Diện tích đất (m2)': DataCleaner.extract_total_area(row_dict),
            'Kích thước mặt tiền (m)': DataCleaner.extract_facade_width(row_dict),
            'Kích thước chiều dài (m)': DataCleaner.extract_land_length(row_dict),
            'Số mặt tiền tiếp giáp': DataCleaner.extract_facade_count(row_dict),
            'Hình dạng': DataCleaner.extract_land_shape(row_dict),
            'Độ rộng ngõ/ngách nhỏ nhất (m)': DataCleaner.extract_alley_width(row_dict),
            'Khoảng cách tới trục đường chính (m)': DataCleaner.extract_distance_to_main_road(row_dict),
            'Mục đích sử dụng đất': 'Đất ở',
            'Yếu tố khác': " | ".join(direct_features) if direct_features else None,
            'Tọa độ (vĩ độ)': row_dict.get('latitude'),
            'Tọa độ (kinh độ)': row_dict.get('longitude'),
            'Hình ảnh của bài đăng': row_dict.get('image_urls'),
            'description': row_dict.get('description')
        }
        cleaned_records.append(processed_data)

    df_cleaned = pd.DataFrame(cleaned_records)

    try:
        # Standardize Province and District using the simplified AddressStandardizer
        address_std = AddressStandardizer(
            config.PROVINCES_SQL_FILE,
            config.DISTRICTS_SQL_FILE
        )
        df_cleaned['Tỉnh/Thành phố'] = df_cleaned['Tỉnh/Thành phố'].apply(address_std.standardize_province)
        df_cleaned['Thành phố/Quận/Huyện/Thị xã'] = df_cleaned['Thành phố/Quận/Huyện/Thị xã'].apply(address_std.standardize_district)
        print("Province and District standardization complete.")
    except FileNotFoundError:
        print("Skipping province/district standardization because data files were not found.")

    df_cleaned['Đường phố'] = df_cleaned['Đường phố'].apply(DataCleaner.validate_and_format_street_name)

    # 1. Drop rows where 'Diện tích đất (m2)' is missing, as it's essential.
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['Diện tích đất (m2)']).reset_index(drop=True)
    rows_after_drop = len(df_cleaned)

    if initial_rows > rows_after_drop:
        print(f"  - Dropped {initial_rows - rows_after_drop} rows with missing 'Diện tích đất (m2)'.")

    # 2. Define helper functions for filling missing values
    def fill_length(row):
        area = row['Diện tích đất (m2)']
        facade = row['Kích thước mặt tiền (m)']

        if pd.isna(facade) or facade == 0:
            return area ** 0.5
        else:
            return area / facade

    def fill_facade(row):
        area = row['Diện tích đất (m2)']
        length = row['Kích thước chiều dài (m)']

        if pd.isna(length) or length == 0:
            return area ** 0.5
        else:
            return area / length

    # 3. Impute missing length ('Kích thước chiều dài (m)')
    mask_length = df_cleaned['Kích thước chiều dài (m)'].isna()

    if mask_length.any():
        print(f"  - Imputing {mask_length.sum()} missing 'Kích thước chiều dài (m)' values...")
        df_cleaned.loc[mask_length, 'Kích thước chiều dài (m)'] = df_cleaned.loc[mask_length].apply(fill_length, axis=1)

    # 4. Impute missing facade width ('Kích thước mặt tiền (m)')
    mask_facade = df_cleaned['Kích thước mặt tiền (m)'].isna()

    if mask_facade.any():
        print(f"  - Imputing {mask_facade.sum()} missing 'Kích thước mặt tiền (m)' values...")
        df_cleaned.loc[mask_facade, 'Kích thước mặt tiền (m)'] = df_cleaned.loc[mask_facade].apply(fill_facade, axis=1)

    # 5. Round the imputed values for cleaner data
    df_cleaned['Kích thước chiều dài (m)'] = df_cleaned['Kích thước chiều dài (m)'].round(2)
    df_cleaned['Kích thước mặt tiền (m)'] = df_cleaned['Kích thước mặt tiền (m)'].round(2)

    initial_rows_before_validation = len(df_cleaned)

    # Create a mask for rows where dimensions are illogical
    mask_invalid_dimensions = (df_cleaned['Kích thước mặt tiền (m)'] > df_cleaned['Diện tích đất (m2)']) | \
                              (df_cleaned['Kích thước chiều dài (m)'] > df_cleaned['Diện tích đất (m2)'])

    # Filter out the invalid rows
    df_cleaned = df_cleaned[~mask_invalid_dimensions].reset_index(drop=True)

    rows_after_validation = len(df_cleaned)
    dropped_count = initial_rows_before_validation - rows_after_validation

    if dropped_count > 0:
        print(f"- Dropped {dropped_count} rows where facade or length > total area.")
    else:
        print("- All dimensions are valid.")

    final_columns = [col for col in config.FINAL_COLUMNS if
                     col not in ['Lợi thế kinh doanh', 'Đơn giá đất', 'Giá ước tính']]
    columns_to_save = final_columns + ['description']  # Add description for saving
    df_final = df_cleaned.reindex(columns=columns_to_save)

    df_final.to_csv(config.CLEANED_DETAILS_OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    print(f"Successfully cleaned and saved {len(df_final)} records to '{config.CLEANED_DETAILS_OUTPUT_FILE}'")


def run_feature_engineering():
    """Step 4: Engineer new features from the cleaned data."""
    if not os.path.exists(config.CLEANED_DETAILS_OUTPUT_FILE):
        print(f"Cleaned data file not found: {config.CLEANED_DETAILS_OUTPUT_FILE}. Run with `--mode clean` first.")
        return

    print(f"Reading cleaned data from '{config.CLEANED_DETAILS_OUTPUT_FILE}'...")
    df = pd.read_csv(config.CLEANED_DETAILS_OUTPUT_FILE)

    print("Engineering new features: 'Giá ước tính', 'Lợi thế kinh doanh', 'Đơn giá đất', 'Tổng diện tích sàn'...")
    df['Giá ước tính'] = df.apply(lambda row: FeatureEngineer.calculate_estimated_price(row.to_dict()), axis=1)
    df['Lợi thế kinh doanh'] = df.apply(lambda row: FeatureEngineer.calculate_business_advantage(row.to_dict()), axis=1)
    df['Tổng diện tích sàn'] = df.apply(lambda row: FeatureEngineer.fill_built_area(row.to_dict()), axis=1)
    df['Đơn giá đất'] = df.apply(lambda row: FeatureEngineer.calculate_land_unit_price(row.to_dict()), axis=1)

    print(f'Giá rao bán/giao dịch NaN {df["Giá rao bán/giao dịch"].isna().sum()} values')
    print(f'Đơn giá xây dựng NaN {df["Đơn giá xây dựng"].isna().sum()} values')
    print(f'Diện tích đất (m2) NaN {df["Diện tích đất (m2)"].isna().sum()} values')
    print(f'Chất lượng còn lại NaN {df["Chất lượng còn lại"].isna().sum()} values')
    print(f'Đơn giá đất NaN {df["Đơn giá đất"].isna().sum()} values')

    initial_rows = len(df)
    df.dropna(subset=['Đơn giá đất'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    rows_after_drop = len(df)
    dropped_count = initial_rows - rows_after_drop
    if dropped_count > 0:
        print(f"  - Dropped {dropped_count} rows with missing 'Đơn giá đất' values.")
    else:
        print("  - No rows dropped due to missing 'Đơn giá đất'.")

    # Ensure the final column order is correct
    df_final = df.reindex(columns=config.FINAL_COLUMNS)

    # Save the final result
    df_final.to_excel(config.FEATURE_ENGINEERED_OUTPUT_FILE, index=False)
    print(
        f"Successfully engineered features and saved {len(df_final)} records to '{config.FEATURE_ENGINEERED_OUTPUT_FILE}'")


def run_ml_imputation():
    """Step 5: Use ML to impute missing 'Độ rộng ngõ/ngách nhỏ nhất (m)' values."""
    if not os.path.exists(config.FEATURE_ENGINEERED_OUTPUT_FILE):
        print(f"Feature engineered file not found: {config.FEATURE_ENGINEERED_OUTPUT_FILE}. Run with `--mode feature` first.")
        return

    print(f"Reading feature-engineered data from '{config.FEATURE_ENGINEERED_OUTPUT_FILE}'...")
    df = pd.read_excel(config.FEATURE_ENGINEERED_OUTPUT_FILE)

    print("Attempting to predict and fill missing alley widths using ML model...")
    df_imputed = _predict_alley_width_ml_step(df)

    # Re-calculate business advantage as it depends on alley width
    print("Re-calculating 'Lợi thế kinh doanh' with imputed alley widths...")
    df_imputed['Lợi thế kinh doanh'] = df_imputed.apply(lambda row: FeatureEngineer.calculate_business_advantage(row.to_dict()), axis=1)

    # Ensure the final column order is correct
    df_final = df_imputed.reindex(columns=config.FINAL_COLUMNS)

    df_final.to_excel(config.ML_IMPUTED_OUTPUT_FILE, index=False)
    print(
        f"Successfully imputed features and saved {len(df_final)} records to '{config.ML_IMPUTED_OUTPUT_FILE}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real‑estate scraper, cleaner & feature-engineering CLI")
    parser.add_argument(
        "--mode",
        choices=["urls", "details", "clean", "feature", "ml"],
        required=True,
        help="'urls' → collect listing URLs\n"
             "'details' → scrape details from URLs\n"
             "'clean' → clean scraped data\n"
             "'feature' → engineer new features from cleaned data\n"
             "'ml' → impute missing alley widths with an ML model"
    )
    args = parser.parse_args()

    if args.mode == "urls":
        run_scrape_urls()
    elif args.mode == "details":
        run_scrape_details()
    elif args.mode == "clean":
        run_cleaning_pipeline()
    elif args.mode == "feature":
        run_feature_engineering()
    elif args.mode == "ml":
        run_ml_imputation()