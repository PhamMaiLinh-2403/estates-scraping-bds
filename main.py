import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src import config
from src.selenium_manager import *
from src.config import *
from src.scraping import Scraper
from src.utils import *
from src.cleaning import DataCleaner, DataImputer, FeatureEngineer
from src.address_standardizer import AddressStandardizer


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


def clean_details():
    """Step 3: Clean the scraped data."""
    if not os.path.exists(config.DETAILS_OUTPUT_FILE):
        print("Details file not found. Run with `--mode details` first.")
        return

    df = pd.read_csv(config.DETAILS_OUTPUT_FILE)
    df.drop_duplicates()
    df.dropna(subset=["title", "description"], inplace=True)

    pattern = r"bán\s*(?:\S+\s+){0,3}?([2-9]|\d{2,})\s+\S*\s*nhà"
    mask = (
        df['title'].str.contains(pattern, case=False, regex=True, na=False) |
        df['description'].str.contains(pattern, case=False, regex=True, na=False)
    )
    df = df[~mask]
    df = df[~df['title'].str.contains('bán dãy nhà', case=False, na=False)]

    print(f"After dropping NaN values and duplicates, there are {len(df)} rows of data in the dataset.")    

    # Initialize Address Standardizer
    address_standardizer = AddressStandardizer(
        provinces_sql_path=PROVINCES_SQL_FILE,
        districts_sql_path=DISTRICTS_SQL_FILE,
        wards_sql_path=WARDS_SQL_FILE,
        streets_sql_path=STREETS_SQL_FILE,
    )

    # 1. Clean and structure the data 
    print("Start extracting and cleaning raw data...")
    df['Tỉnh/Thành phố'] = df.apply(DataCleaner.extract_city, axis=1)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(DataCleaner.extract_district, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(DataCleaner.extract_ward, axis=1)
    df['Đường phố'] = df.apply(DataCleaner.extract_street, axis=1)
    df['Chi tiết'] = df.apply(DataCleaner.extract_address_details, axis=1)
    df['Thời điểm giao dịch/rao bán'] = df['main_info'].apply(DataCleaner.extract_published_date)
    df['Giá rao bán/giao dịch'] = df.apply(DataCleaner.extract_price, axis=1)
    df['Số tầng công trình'] = df.apply(DataCleaner.extract_num_floors, axis=1)
    df['Số mặt tiền tiếp giáp'] = df.apply(DataCleaner.extract_facade_count, axis=1)
    df['Hình dạng'] = df.apply(DataCleaner.extract_land_shape, axis=1)
    df['Chất lượng còn lại'] = df.apply(DataCleaner.estimate_remaining_quality, axis=1)
    df['Đơn giá xây dựng'] = df.apply(DataCleaner.extract_construction_cost, axis=1)
    df['Diện tích đất (m2)'] = df.apply(DataCleaner.extract_total_area, axis=1)
    df['Kích thước mặt tiền (m)'] = df.apply(DataCleaner.extract_width, axis=1)
    df['Kích thước chiều dài (m)'] = df.apply(DataCleaner.extract_length, axis=1)
    df['Mục đích sử dụng đất'] = df.apply(DataCleaner.extract_land_use, axis=1)
    df['Diện tích xây dựng'] = df.apply(DataCleaner.extract_construction_area, axis=1)
    df['Tổng diện tích sàn'] = df.apply(DataCleaner.extract_building_area, axis=1)
    df['Độ rộng ngõ/ngách nhỏ nhất (m)'] = df.apply(DataCleaner.extract_adjacent_lane_width, axis=1)
    df['Khoảng cách tới trục đường chính (m)'] = df.apply(DataCleaner.extract_distance_to_the_main_road, axis=1)
    df['description'] = df['description'].apply(DataCleaner.clean_description_text)
    df['title'] = df['title']
    
    # 2. Standardize addresses 
    print("Standardizing addresses...")
    df['Tỉnh/Thành phố'] = df['Tỉnh/Thành phố'].apply(address_standardizer.standardize_province)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(address_standardizer.standardize_district, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(address_standardizer.standardize_ward, axis=1)

    # 3. Imputing missing values
    print("Start imputing missing values...")
    df = DataImputer.fill_missing_width(df)
    df['Kích thước chiều dài (m)'] = df.apply(DataImputer.fill_missing_length, axis=1)

    # 4. Create new features 
    print("Start feature engineeering...")
    df['Giá ước tính'] = df.apply(FeatureEngineer.calculate_estimated_price, axis=1)
    df['Lợi thế kinh doanh'] = df.apply(FeatureEngineer.calculate_business_advantage, axis=1)
    df['Đơn giá đất'] = df.apply(FeatureEngineer.calculate_land_unit_price, axis=1)

    # 5. Structure the data into a final df
    print("Start structuring the data...")
    final_df = pd.DataFrame()
    final_df['Tỉnh/Thành phố'] = df['Tỉnh/Thành phố']
    final_df['Thành phố/Quận/Huyện/Thị xã'] = df['Thành phố/Quận/Huyện/Thị xã']
    final_df['Xã/Phường/Thị trấn'] = df['Xã/Phường/Thị trấn']
    final_df['Đường phố'] = df['Đường phố']
    final_df['Chi tiết'] = df['Chi tiết']
    final_df['Nguồn thông tin'] = df["url"]
    final_df['Tình trạng giao dịch'] = 'Đang rao bán' 
    final_df['Thời điểm giao dịch/rao bán'] = df['Thời điểm giao dịch/rao bán']
    final_df['Thông tin liên hệ'] = None  
    final_df['Giá rao bán/giao dịch'] = df['Giá rao bán/giao dịch']
    final_df['Giá ước tính'] = df['Giá ước tính']
    final_df['Loại đơn giá (đ/m2 hoặc đ/m ngang)'] = 'đ/m2'  
    final_df['Đơn giá đất'] = df['Đơn giá đất']
    final_df['Lợi thế kinh doanh'] = df['Lợi thế kinh doanh']
    final_df['Số tầng công trình'] = df['Số tầng công trình']
    final_df['Tổng diện tích sàn'] = df['Tổng diện tích sàn']
    final_df['Đơn giá xây dựng'] = df['Đơn giá xây dựng']
    final_df['Năm xây dựng'] = None 
    final_df['Chất lượng còn lại'] = df['Chất lượng còn lại']
    final_df['Diện tích đất (m2)'] = df['Diện tích đất (m2)']
    final_df['Kích thước mặt tiền (m)'] = df['Kích thước mặt tiền (m)']
    final_df['Kích thước chiều dài (m)'] = df['Kích thước chiều dài (m)']
    final_df['Số mặt tiền tiếp giáp'] = df['Số mặt tiền tiếp giáp']
    final_df['Hình dạng'] = df['Hình dạng']
    final_df['Độ rộng ngõ/ngách nhỏ nhất (m)'] = df['Độ rộng ngõ/ngách nhỏ nhất (m)']
    final_df['Khoảng cách tới trục đường chính (m)'] = df['Khoảng cách tới trục đường chính (m)']
    final_df['Mục đích sử dụng đất'] = df['Mục đích sử dụng đất']
    final_df['Yếu tố khác'] = df['description']
    final_df['Tọa độ (vĩ độ)'] = df['latitude']
    final_df['Tọa độ (kinh độ)'] = df['longitude']
    final_df['Hình ảnh của bài đăng'] = df['image_urls']

    subset = ['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố',
    'Chi tiết', 'Thời điểm giao dịch/rao bán', 'Giá rao bán/giao dịch', 'Số tầng công trình',
    'Số mặt tiền tiếp giáp', 'Hình dạng', 'Chất lượng còn lại', 'Đơn giá xây dựng',
    'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Mục đích sử dụng đất',
    'Tổng diện tích sàn', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)',
    'Giá ước tính', 'Lợi thế kinh doanh', 'Đơn giá đất', 'Nguồn thông tin', 
    'Tọa độ (vĩ độ)', 'Tọa độ (kinh độ)'
]

    final_df.dropna(subset=subset, inplace=True)
    
    # Export the cleaned data 
    final_df.to_excel(config.CLEANED_DETAILS_OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(final_df)} cleaned rows into {config.CLEANED_DETAILS_OUTPUT_FILE}")

def clean_details_for_land():
    """Step 3: Clean the data (for land )"""
    if not os.path.exists(config.DETAILS_OUTPUT_FILE):
        print("Details file not found. Run with `--mode details` first.")
        return

    df = pd.read_csv(config.DETAILS_OUTPUT_FILE)
    df.drop_duplicates()
    df.dropna(subset=["title", "description"], inplace=True)

    pattern = r"bán\s+\w+\s+(lô đất|mảnh đất)"

    df = df[
        ~(
            df['description'].str.contains(pattern, case=False, regex=True, na=False) |
            df['title'].str.contains(pattern, case=False, regex=True, na=False)
        )
    ]

    print(f"After dropping NaN values and duplicates, there are {len(df)} rows of data in the dataset.")    

    # Initialize Address Standardizer
    address_standardizer = AddressStandardizer(
        provinces_sql_path=PROVINCES_SQL_FILE,
        districts_sql_path=DISTRICTS_SQL_FILE,
        wards_sql_path=WARDS_SQL_FILE,
        streets_sql_path=STREETS_SQL_FILE,
    )

    # 1. Clean and structure the data 
    print("Start extracting and cleaning raw data...")
    df['Tỉnh/Thành phố'] = df.apply(DataCleaner.extract_city, axis=1)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(DataCleaner.extract_district, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(DataCleaner.extract_ward, axis=1)
    df['Đường phố'] = df.apply(DataCleaner.extract_street, axis=1)
    df['Chi tiết'] = df.apply(DataCleaner.extract_address_details, axis=1)
    df['Thời điểm giao dịch/rao bán'] = df['main_info'].apply(DataCleaner.extract_published_date)
    df['Giá rao bán/giao dịch'] = df.apply(DataCleaner.extract_price, axis=1)
    df['Số tầng công trình'] = ""
    df['Số mặt tiền tiếp giáp'] = df.apply(DataCleaner.extract_facade_count, axis=1)
    # df['Hình dạng'] = df.apply(DataCleaner.extract_land_shape, axis=1)
    df['Chất lượng còn lại'] = ""
    df['Đơn giá xây dựng'] = ""
    df['Diện tích đất (m2)'] = df.apply(DataCleaner.extract_total_area, axis=1)
    df['Kích thước mặt tiền (m)'] = df.apply(DataCleaner.extract_width, axis=1)
    df['Kích thước chiều dài (m)'] = df.apply(DataCleaner.extract_length, axis=1)
    # df['Mục đích sử dụng đất'] = df.apply(DataCleaner.extract_land_use, axis=1)
    df['Tổng diện tích sàn'] = ""
    df['Độ rộng ngõ/ngách nhỏ nhất (m)'] = df.apply(DataCleaner.extract_adjacent_lane_width, axis=1)
    df['Khoảng cách tới trục đường chính (m)'] = df.apply(DataCleaner.extract_distance_to_the_main_road, axis=1)
    df['description'] = df['description'].apply(DataCleaner.clean_description_text)
    df['title'] = df['title']

    # 2. Standardize addresses 
    print("Standardizing addresses...")
    df['Tỉnh/Thành phố'] = df['Tỉnh/Thành phố'].apply(address_standardizer.standardize_province)
    df['Thành phố/Quận/Huyện/Thị xã'] = df.apply(address_standardizer.standardize_district, axis=1)
    df['Xã/Phường/Thị trấn'] = df.apply(address_standardizer.standardize_ward, axis=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real‑estate scraper, cleaner & feature-engineering CLI")
    parser.add_argument(
        "--mode",
        choices=["urls", "details", "clean", "clean_land"],
        required=True,
        help="'urls' → collect listing URLs\n"
             "'details' → scrape details from URLs\n"
             "'clean' → clean scraped data\n"
             "'clean_land' → clean scraped data for land listings"
    )
    args = parser.parse_args()

    if args.mode == "urls":
        run_scrape_urls()
    elif args.mode == "details":
        run_scrape_details()
    elif args.mode == "clean":
        clean_details()
    elif args.mode == "clean_land":
        clean_details_for_land()