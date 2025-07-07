import argparse
import csv
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from src.selenium_manager import create_stealth_driver
from src.scraping import Scraper
from src.cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src import config


# --- File I/O ---
def save_urls_to_csv(urls, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        writer.writerows([[u] for u in urls])

    print(f"✅  Saved {len(urls)} URLs → {file_path}")


def save_details_to_csv(details, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(details).to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅  Saved {len(details)} listing records → {file_path}")


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
    print(f"[Worker {worker_id}]  Sleeping {start_delay:.1f}s before start.")
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
        print("🚫  URL file not found. Run with `--mode urls` first.")
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
        print(f"🚫  Raw details file not found: {config.DETAILS_OUTPUT_FILE}. Run with `--mode details` first.")
        return

    print(f"Reading raw data from '{config.DETAILS_OUTPUT_FILE}'...")
    df_raw = pd.read_csv(config.DETAILS_OUTPUT_FILE)
    print(f"Processing {len(df_raw)} records...")

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
            'Hình ảnh của bài đăng': row_dict.get('image_urls')
        }
        cleaned_records.append(processed_data)

    df_cleaned = pd.DataFrame(cleaned_records)
    final_columns = [col for col in config.FINAL_COLUMNS if col not in ['Lợi thế kinh doanh', 'Đơn giá đất']]
    df_final = df_cleaned.reindex(columns=final_columns)

    df_final.to_csv(config.CLEANED_DETAILS_OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Successfully cleaned and saved {len(df_final)} records to '{config.CLEANED_DETAILS_OUTPUT_FILE}'")


def run_feature_engineering():
    """Step 4: Engineer new features from the cleaned data."""
    if not os.path.exists(config.CLEANED_DETAILS_OUTPUT_FILE):
        print(f"🚫 Cleaned data file not found: {config.CLEANED_DETAILS_OUTPUT_FILE}. Run with `--mode clean` first.")
        return

    print(f"Reading cleaned data from '{config.CLEANED_DETAILS_OUTPUT_FILE}'...")
    df = pd.read_csv(config.CLEANED_DETAILS_OUTPUT_FILE)

    print("Engineering new features...")

    # Use .apply to generate the new columns
    df['Lợi thế kinh doanh'] = df.apply(lambda row: FeatureEngineer.calculate_business_advantage(row.to_dict()), axis=1)
    df['Đơn giá đất'] = df.apply(lambda row: FeatureEngineer.calculate_land_unit_price(row.to_dict()), axis=1)

    # Ensure the final column order is correct
    df_final = df.reindex(columns=config.FINAL_COLUMNS)

    # Save the final result
    df_final.to_csv(config.FEATURE_ENGINEERED_OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
    print(
        f"✅ Successfully engineered features and saved {len(df_final)} records to '{config.FEATURE_ENGINEERED_OUTPUT_FILE}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real‑estate scraper, cleaner & feature-engineering CLI")
    parser.add_argument(
        "--mode",
        choices=["urls", "details", "clean", "feature"],
        required=True,
        help="'urls' → collect listing URLs\n"
             "'details' → scrape details from URLs\n"
             "'clean' → clean scraped data\n"
             "'feature' → engineer new features from cleaned data"
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