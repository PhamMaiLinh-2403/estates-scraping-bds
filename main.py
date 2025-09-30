import argparse
import os
from datetime import date
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

from src.selenium_manager import create_stealth_driver
from src.scraping import Scraper
from src.cleaning import DataCleaner, drop_mixed_listings, is_land_only
from src.feature_engineering import FeatureEngineer
from src.address_standardizer import AddressStandardizer
from src import config
from src.utils import save_urls_to_csv, save_details_to_csv, chunks
from src.tasks import scrape_worker
from src.modelling import predict_alley_width


# --- Pipeline Steps ---
def run_scrape_urls():
    """Step 1: Scrape listing URLs from search pages."""
    driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
    scraper = Scraper(driver)
    urls = scraper.scrape_listing_urls(config.SEARCH_PAGE_URL, config.PAGE_NUMBER)
    save_urls_to_csv(urls, config.URLS_OUTPUT_FILE)
    driver.quit()


def run_scrape_details():
    """Step 2: Scrape detailed information for each URL."""
    if not os.path.exists(config.URLS_OUTPUT_FILE):
        print("URL file not found. Run with `--mode urls` first.")
        return

    urls = pd.read_csv(config.URLS_OUTPUT_FILE)["url"].tolist()
    print(f"Loaded {len(urls)} URLs for detail‑scrape.")

    # Load existing record IDs to prevent duplicates
    existing_ids = set()
    if config.SCRAPING_DETAILS_CONFIG["append_mode"] and os.path.exists(config.DETAILS_OUTPUT_FILE):
        try:
            print(f"Loading existing data from {config.DETAILS_OUTPUT_FILE} to prevent duplicates...")
            df_existing = pd.read_csv(config.DETAILS_OUTPUT_FILE, usecols=['id'], on_bad_lines='skip')
            # Normalize IDs for comparison
            existing_ids = set(df_existing['id'].dropna().astype(str).str.replace(r'\.0$', '', regex=True))
            print(f"Found {len(existing_ids)} existing listing IDs.")
        except Exception as e:
            print(f"Warning: Could not read existing details file to check for duplicates. Error: {e}")

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
    stop_event = threading.Event()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(scrape_worker, wid, subset, existing_ids, stop_event): wid
                for wid, subset in enumerate(url_chunks)
            }

            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    worker_details = fut.result()
                    if worker_details:
                        details_all.extend(worker_details)
                    print(f"[Worker {wid}]: {len(worker_details or [])} listings scraped.")
                except Exception as exc:
                    print(f"[Worker {wid}]: raised {exc!r}")

    except KeyboardInterrupt:
        print("\n Scraping interrupted by user (Ctrl+C).")
        print("Notifying all workers to shut down gracefully...")
        stop_event.set()

        for fut, wid in futures.items():
            if fut.done():
                try:
                    worker_details = fut.result()
                    if worker_details:
                        details_all.extend(worker_details)
                    print(f"[Worker {wid}]: (post-interrupt) {len(worker_details or [])} listings scraped.")
                except Exception as exc:
                    print(f"[Worker {wid}]: (post-interrupt) raised {exc!r}")
            else:
                print(f"[Worker {wid}]: still running — results not collected.")

    finally:
        if details_all:
            save_details_to_csv(details_all, config.DETAILS_OUTPUT_FILE)
            print(f"\nSaved {len(details_all)} listings before exit.")
        else:
            print("\nNo data collected to save.")


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

        # --- 1. Detect if it is a land-only property ---
        is_land = is_land_only(row_dict)

        # --- 2. Extract all data ---
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
            'description': row_dict.get('description'),
            'is_land': is_land  # <-- Add the temporary flag here
        }

        # --- 3. Apply special logic if it's land only ---
        if is_land:
            processed_data['Số tầng công trình'] = 0
            processed_data['Đơn giá xây dựng'] = 0
            processed_data['Tổng diện tích sàn'] = 0
            processed_data['Chất lượng còn lại'] = 0
        
        cleaned_records.append(processed_data)

    df_cleaned = pd.DataFrame(cleaned_records)

    try:
        # Standardize Province and District using the simplified AddressStandardizer
        address_std = AddressStandardizer(
            config.PROVINCES_SQL_FILE,
            config.DISTRICTS_SQL_FILE,
            config.WARDS_SQL_FILE,
            config.STREETS_SQL_FILE
        )
        df_cleaned['Tỉnh/Thành phố'] = df_cleaned['Tỉnh/Thành phố'].apply(address_std.standardize_province)
        df_cleaned['short_address'] = df_raw['short_address']
        # df_cleaned['Thành phố/Quận/Huyện/Thị xã'] = df_cleaned['Thành phố/Quận/Huyện/Thị xã'].apply(address_std.standardize_district)
        df_cleaned['Thành phố/Quận/Huyện/Thị xã'] = df_cleaned.apply(address_std.standardize_district, axis=1)
        df_cleaned['Xã/Phường/Thị trấn'] = df_cleaned.apply(address_std.standardize_ward, axis=1)
        df_cleaned.drop(columns=['short_address'], inplace=True)
        print("Province and District standardization complete.")
    except FileNotFoundError:
        print("Skipping province/district standardization because data files were not found.")

    df_cleaned['Đường phố'] = df_cleaned['Đường phố'].apply(DataCleaner.validate_and_format_street_name)

    # 1. Drop rows where 'Diện tích đất (m2)' is missing, as it's essential. The missing value might be written as empty strings.
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['Diện tích đất (m2)']).reset_index(drop=True)
    rows_after_drop = len(df_cleaned)

    if initial_rows > rows_after_drop:
        print(f"- Dropped {initial_rows - rows_after_drop} rows with missing 'Diện tích đất (m2)'.")

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
        print(f"- Imputing {mask_length.sum()} missing 'Kích thước chiều dài (m)' values...")
        df_cleaned.loc[mask_length, 'Kích thước chiều dài (m)'] = df_cleaned.loc[mask_length].apply(fill_length, axis=1)

    # 4. Impute missing facade width ('Kích thước mặt tiền (m)')
    mask_facade = df_cleaned['Kích thước mặt tiền (m)'].isna()

    if mask_facade.any():
        print(f"- Imputing {mask_facade.sum()} missing 'Kích thước mặt tiền (m)' values...")
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

    df_final.to_csv(config.CLEANED_DETAILS_OUTPUT_FILE, index=False)
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

    initial_rows = len(df)
    df.dropna(subset=['Đơn giá đất'], inplace=True)
    df = df[df["Giá rao bán/giao dịch"] >= 100_000_000]
    df.reset_index(drop=True, inplace=True)

    rows_after_drop = len(df)
    dropped_count = initial_rows - rows_after_drop
    if dropped_count > 0:
        print(f"- Dropped {dropped_count} rows with missing 'Đơn giá đất' values.")
    else:
        print("- No rows dropped due to missing 'Đơn giá đất'.")

    print(f'Processing time columns...')
    df.dropna(subset=['Thời điểm giao dịch/rao bán'], inplace=True)
    df['Thời điểm giao dịch/rao bán'] = pd.to_datetime(df['Thời điểm giao dịch/rao bán'], format='%d/%m/%Y')
    today = date.today()
    df['Số ngày tính từ lúc đăng tin'] = df['Thời điểm giao dịch/rao bán'].apply(lambda x: (today - x.date()).days)

    # Ensure the final column order is correct
    FE_COLUMNS = config.FINAL_COLUMNS
    FE_COLUMNS.append('Số ngày tính từ lúc đăng tin')
    df_final = df.reindex(columns=FE_COLUMNS)

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
    df_imputed = predict_alley_width(df)

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