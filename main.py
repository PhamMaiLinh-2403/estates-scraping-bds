import argparse
import os
import numpy as np
import pandas as pd
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed 

from src.config import * 
from src.selenium_manager import *
from src.utils import *
from src.cleaning import DataCleaner, DataImputer, FeatureEngineer, LandCleaner
from src.address_standardizer import AddressStandardizer
from src.scraping import Scraper


# --- DATA SCHEMA ---
FINAL_SCHEMA = {
    "Tỉnh/Thành phố": "Tỉnh/Thành phố",
    "Thành phố/Quận/Huyện/Thị xã": "Thành phố/Quận/Huyện/Thị xã",
    "Xã/Phường/Thị trấn": "Xã/Phường/Thị trấn",
    "Đường phố": "Đường phố",
    "Chi tiết": "Chi tiết",
    "Nguồn thông tin": "url",
    "Tình trạng giao dịch": "status_const",
    "Thời điểm giao dịch/rao bán": "Thời điểm giao dịch/rao bán",
    "Thông tin liên hệ": "contact_const",
    "Giá rao bán/giao dịch": "Giá rao bán/giao dịch",
    "Giá ước tính": "Giá ước tính",
    "Loại đơn giá (đ/m2 hoặc đ/m ngang)": "unit_type_const",
    "Đơn giá đất": "Đơn giá đất",
    "Lợi thế kinh doanh": "Lợi thế kinh doanh",
    "Số tầng công trình": "Số tầng công trình",
    "Tổng diện tích sàn": "Tổng diện tích sàn",
    "Đơn giá xây dựng": "Đơn giá xây dựng",
    "Năm xây dựng": "year_const",
    "Chất lượng còn lại": "Chất lượng còn lại",
    "Diện tích đất (m2)": "Diện tích đất (m2)",
    "Kích thước mặt tiền (m)": "Kích thước mặt tiền (m)",
    "Kích thước chiều dài (m)": "Kích thước chiều dài (m)",
    "Số mặt tiền tiếp giáp": "Số mặt tiền tiếp giáp",
    "Hình dạng": "Hình dạng",
    "Độ rộng ngõ/ngách nhỏ nhất (m)": "Độ rộng ngõ/ngách nhỏ nhất (m)",
    "Khoảng cách tới trục đường chính (m)": "Khoảng cách tới trục đường chính (m)",
    "Mục đích sử dụng đất": "Mục đích sử dụng đất",
    "Yếu tố khác": "description",
    "Tọa độ (vĩ độ)": "latitude",
    "Tọa độ (kinh độ)": "longitude",
    "Hình ảnh của bài đăng": "image_urls",
}


# --- SCRAPING FUNCTIONS ---
def run_scrape_urls():
    """Step 1: Scrape listing URLs from search pages (Parallelized)."""
    
    start_page = config.START_PAGE_NUMBER
    end_page = config.END_PAGE_NUMBER
    max_workers = config.MAX_WORKERS
    search_url = config.SEARCH_PAGE_URL

    print(f"Preparing to scrape URLs from page {start_page} to {end_page} with {max_workers} workers.")

    # 1. Calculate Ranges
    page_ranges = split_page_ranges(start_page, end_page, max_workers)
    
    if not page_ranges:
        print("Invalid page range or worker count.")
        return

    print(f"Work distribution: {page_ranges}")

    # 2. Setup Multiprocessing
    manager = multiprocessing.Manager()
    stop_event = manager.Event()
    all_urls = []

    try:
        with ProcessPoolExecutor(max_workers=len(page_ranges)) as pool:
            futures = {
                pool.submit(
                    scrape_urls_worker, 
                    wid, 
                    search_url, 
                    r_start, 
                    r_end, 
                    stop_event
                ): wid
                for wid, (r_start, r_end) in enumerate(page_ranges)
            }

            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    urls = fut.result()
                    if urls:
                        all_urls.extend(urls)
                    print(f"[Main] Worker {wid} returned {len(urls)} URLs.")
                except Exception as exc:
                    print(f"[Main] Worker {wid} generated an exception: {exc}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted! Stopping workers...")
        stop_event.set()
    
    # 3. Save Results
    # Remove duplicates just in case
    unique_urls = list(set(all_urls))
    print(f"Total unique URLs found: {len(unique_urls)}")
    save_urls_to_csv(unique_urls, config.URLS_OUTPUT_FILE)


def run_scrape_details():
    """Step 2: Scrape detailed information for each URL (Parallelized) with Temp File Merging."""
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
            existing_ids = set(df_existing['id'].dropna().astype(str).str.replace(r'\.0$', '', regex=True))
            print(f"Found {len(existing_ids)} existing listing IDs.")
        except Exception as e:
            print(f"Warning: Could not read existing details file. {e}")

    start = config.SCRAPING_DETAILS_CONFIG["start_index"]
    count = config.SCRAPING_DETAILS_CONFIG["count"]
    urls_to_scrape = urls[start: start + count] if count else urls[start:]

    max_workers = min(config.MAX_WORKERS, len(urls_to_scrape))
    if max_workers == 0:
        print("No URLs to scrape.")
        return

    url_chunks = list(chunks(urls_to_scrape, max_workers))
    print(f"Spawning {max_workers} workers.")
    
    # Ensure Temp Directory Exists
    os.makedirs(config.TEMP_OUTPUT_DIR, exist_ok=True)
    
    manager = multiprocessing.Manager()
    stop_event = manager.Event()

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            # Pass temp_dir path to workers
            futures = {
                pool.submit(
                    scrape_worker, 
                    wid, 
                    subset, 
                    existing_ids, 
                    stop_event, 
                    config.TEMP_OUTPUT_DIR
                ): wid
                for wid, subset in enumerate(url_chunks)
            }

            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    temp_file = fut.result()
                    print(f"[Main] Worker {wid} finished writing to {temp_file}.")
                except Exception as exc:
                    print(f"[Main] Worker {wid} raised {exc!r}")

    except KeyboardInterrupt:
        print("\n[Main] Scraping interrupted by user (Ctrl+C). Stopping workers...")
        stop_event.set()
        # Allow a moment for workers to close files safely if possible
        pass

    finally:
        print("\n[Main] Merging temporary files...")
        merge_temp_files(
            config.TEMP_OUTPUT_DIR, 
            config.DETAILS_OUTPUT_FILE, 
            append_mode=config.SCRAPING_DETAILS_CONFIG["append_mode"]
        )
        print("[Main] Process finished.")


# --- CLEANING FUNCTIONS ---
def run_cleaning_pipeline(mode="house"):
    """Step 3: Full cleaning pipeline for House and Land."""
    if not os.path.exists(config.DETAILS_OUTPUT_FILE):
        return print("Details file not found.")

    print(f"Starting {mode} cleaning pipeline...")
    df = pd.read_csv(config.DETAILS_OUTPUT_FILE).drop_duplicates()
    df.dropna(subset=["title", "description"], inplace=True)

    # 1. Filtering
    if mode == "house":
        pattern = r"bán\s*(?:\S+\s+){0,3}?([2-9]|\d{2,})\s+\S*\s*nhà"
        mask = df['title'].str.contains(pattern, case=False, regex=True, na=False) | \
               df['description'].str.contains(pattern, case=False, regex=True, na=False)
        df = df[~mask]
        df = df[~df['title'].str.contains('bán dãy nhà', case=False, na=False)]

    else: 
        pattern = r"bán\s+\w+\s+(lô|mảnh)"
        df = df[~(df['description'].str.contains(pattern, case=False, regex=True, na=False) | 
                  df['title'].str.contains(pattern, case=False, regex=True, na=False))]
        df['land_category'] = df.apply(LandCleaner.categorize_lands, axis=1)
        df = df[df['land_category'] != 2]

    print(f"After dropping NaN values and duplicates, there are {len(df)} rows of data in the dataset.")

    # 2. Extraction & Cleaning
    # Initialize standardizer 
    standardizer = AddressStandardizer(
        config.PROVINCES_SQL_FILE, config.DISTRICTS_SQL_FILE, 
        config.WARDS_SQL_FILE, config.STREETS_SQL_FILE
    )

    print("Extracting raw data...")
    df['Tỉnh/Thành phố'] = df.apply(DataCleaner.extract_city, axis=1).apply(standardizer.standardize_province)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(DataCleaner.extract_district, axis=1)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(standardizer.standardize_district, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(DataCleaner.extract_ward, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(standardizer.standardize_ward, axis=1)
    
    df['Đường phố'] = df.apply(DataCleaner.extract_street, axis=1)
    df['Chi tiết'] = df.apply(DataCleaner.extract_address_details, axis=1)
    df['Thời điểm giao dịch/rao bán'] = df['main_info'].apply(DataCleaner.extract_published_date)
    df['Giá rao bán/giao dịch'] = df.apply(DataCleaner.extract_price, axis=1)
    df['Số mặt tiền tiếp giáp'] = df.apply(DataCleaner.extract_facade_count, axis=1)
    df['Diện tích đất (m2)'] = df.apply(DataCleaner.extract_total_area, axis=1)
    df['Kích thước mặt tiền (m)'] = df.apply(DataCleaner.extract_width, axis=1)
    df['Độ rộng ngõ/ngách nhỏ nhất (m)'] = df.apply(DataCleaner.extract_adjacent_lane_width, axis=1)
    df['Khoảng cách tới trục đường chính (m)'] = df.apply(DataCleaner.extract_distance_to_the_main_road, axis=1)
    df['description'] = df['description'].apply(DataCleaner.clean_description_text)

    # Mode-specific column logic
    if mode == "house":
        df['Số tầng công trình'] = df.apply(DataCleaner.extract_num_floors, axis=1)
        df['Hình dạng'] = df.apply(DataCleaner.extract_land_shape, axis=1)
        df['Chất lượng còn lại'] = df.apply(DataCleaner.estimate_remaining_quality, axis=1)
        df['Đơn giá xây dựng'] = df.apply(DataCleaner.extract_construction_cost, axis=1)
        df['Mục đích sử dụng đất'] = df.apply(DataCleaner.extract_land_use, axis=1)
        df['Tổng diện tích sàn'] = df.apply(DataCleaner.extract_building_area, axis=1)
    else:
        df['Số tầng công trình'] = np.where(df["land_category"] == 0, 0, 1)
        df['Hình dạng'] = df.apply(LandCleaner.get_land_shape, axis=1)
        df['Chất lượng còn lại'] = np.where(df["land_category"] == 1, df.apply(DataCleaner.estimate_remaining_quality, axis=1), 0)
        df['Đơn giá xây dựng'] = np.where(df["land_category"] == 1, 4_000_000, 0)
        df['Mục đích sử dụng đất'] = df.apply(LandCleaner.get_land_use, axis=1)
        df['Tổng diện tích sàn'] = np.where(df["land_category"] == 1, df.apply(DataCleaner.extract_building_area, axis=1), 0)

    # 3. Imputation & Feature Engineering
    print("Running imputation and feature engineering...")
    df = DataImputer.fill_missing_width(df)
    df['Kích thước chiều dài (m)'] = df.apply(DataImputer.fill_missing_length, axis=1)
    df['Giá ước tính'] = df.apply(FeatureEngineer.calculate_estimated_price, axis=1)
    df['Lợi thế kinh doanh'] = df.apply(FeatureEngineer.calculate_business_advantage, axis=1)
    df['Đơn giá đất'] = df.apply(FeatureEngineer.calculate_land_unit_price, axis=1)

    # 4. Final Formatting using Schema
    print("Finalizing structure...")
    df['status_const'] = 'Đang rao bán'
    df['contact_const'] = None
    df['unit_type_const'] = 'đ/m2'
    df['year_const'] = None
    
    # Rename and filter columns based on pre-defined schema 
    final_df = df.rename(columns={v: k for k, v in FINAL_SCHEMA.items() if v in df.columns})
    final_df = final_df[list(FINAL_SCHEMA.keys())]

    # Required subset for dropna
    subset = [
    'Tỉnh/Thành phố',
    'Thành phố/Quận/Huyện/Thị xã',
    'Xã/Phường/Thị trấn',
    'Đường phố',
    'Chi tiết',
    'Nguồn thông tin', 
    'Thời điểm giao dịch/rao bán',
    'Giá rao bán/giao dịch',
    'Giá ước tính',
    'Đơn giá đất',
    'Lợi thế kinh doanh',
    'Số tầng công trình', 
    'Tổng diện tích sàn', 
    'Đơn giá xây dựng',
    'Chất lượng còn lại',
    'Diện tích đất (m2)',
    'Kích thước mặt tiền (m)',
    'Kích thước chiều dài (m)',
    'Số mặt tiền tiếp giáp',
    'Hình dạng',
    'Độ rộng ngõ/ngách nhỏ nhất (m)',
    'Khoảng cách tới trục đường chính (m)',
    'Mục đích sử dụng đất',
    'Tọa độ (vĩ độ)',
    'Tọa độ (kinh độ)',
]
    # Drop duplicates
    old_size = final_df.shape[0]
    final_df.drop_duplicates(subset=['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố', 'Giá rao bán/giao dịch', 'Giá ước tính', 'Đơn giá đất', 'Lợi thế kinh doanh', 'Số tầng công trình', 'Tổng diện tích sàn', 'Đơn giá xây dựng', 'Chất lượng còn lại', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất'], inplace=True)
    print(f'Dropped {old_size - final_df.shape[0]} duplicates')

    # Save file without dropping NaN values
    final_df['Time'] = pd.to_datetime(final_df['Thời điểm giao dịch/rao bán'], dayfirst=True)
    min_date = final_df['Time'].min().strftime('%d.%m.%Y')
    max_date = final_df['Time'].max().strftime('%d.%m.%Y')
    final_df.drop(columns='Time', inplace=True)
    OUTPUT_NAME = f"{OUTPUT_DIR}/{min_date}-{max_date}.xlsx"
    OUTPUT_NAME_RAW = f"{OUTPUT_DIR}/{min_date}-{max_date}_raw.xlsx"
    final_df.to_excel(OUTPUT_NAME_RAW, index=False)
    print(f"Cleaned {len(final_df)} rows. Saved to {OUTPUT_NAME_RAW}")

    final_df.dropna(subset=subset, inplace=True)
    final_df.to_excel(OUTPUT_NAME, index=False)
    date_info = {"start_time": min_date, "end_time": max_date, "file_dir": OUTPUT_NAME}
    with open('output/Batdongsan/date.jsonl', 'a') as f:
        f.write(json.dumps(date_info) + '\n')
        
    # final_df.to_excel(config.CLEANED_DETAILS_OUTPUT_FILE, index=False)
    print(f"Cleaned {len(final_df)} rows. Saved to {OUTPUT_NAME}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-estate Tool")
    parser.add_argument("--mode", required=True, choices=["urls", "details", "clean", "clean_land"])
    args = parser.parse_args()

    if args.mode == "urls":
        run_scrape_urls()
    elif args.mode == "details":
        run_scrape_details()
    elif args.mode == "clean":
        run_cleaning_pipeline(mode="house")
    elif args.mode == "clean_land":
        run_cleaning_pipeline(mode="land")