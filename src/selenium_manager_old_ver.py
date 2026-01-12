import time 
import random
import threading
import os
from seleniumbase import Driver

from . import config
from .scraping import Scraper
from .utils import save_batch

def create_stealth_driver(headless: bool = True) -> Driver:
    """
    Creates and returns a "supercharged" Selenium driver instance using seleniumbase's UC mode.
    """
    driver = Driver(
        uc=config.SELENIUM_CONFIG["uc_driver"],
        headless=headless,
        agent=None,
    )

    width, height = map(int, config.SELENIUM_CONFIG["window_size"].split(','))
    driver.set_window_size(width, height)

    return driver

def scrape_worker(worker_id: int, url_subset: list[str], existing_ids: set[str], stop_event: threading.Event, temp_dir: str) -> str:
    """
    Defines the task for a single scraping worker.
    Writes incrementally to a temp file managed by worker_id.
    Returns the path to the temp file.
    """
    base = config.SCRAPING_DETAILS_CONFIG.get("stagger_step_sec", 2.0)
    start_delay = worker_id * base
    print(f"[Worker {worker_id}]: Sleeping {start_delay:.1f}s before start.")
    time.sleep(start_delay)

    driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
    scraper = Scraper(driver)
    
    # Define temp file path
    worker_file_name = f"worker_{worker_id}.csv"
    worker_file_path = temp_dir / worker_file_name
    
    batch_size = config.SCRAPING_DETAILS_CONFIG.get("batch_size", 10)
    batch_buffer = []

    try:
        for idx, url in enumerate(url_subset, 1):
            if stop_event.is_set():
                break

            print(f"[Worker {worker_id}]  {idx}/{len(url_subset)}  → {url}")
            data = scraper.scrape_listing_details(url)

            if data:
                listing_id = str(data.get("id")).replace(".0", "") 
                if listing_id in existing_ids:
                    print(f"[Worker {worker_id}] Skipping already-scraped ID: {listing_id}")
                    continue
                
                batch_buffer.append(data)

            # Flush batch to disk
            if len(batch_buffer) >= batch_size:
                save_batch(batch_buffer, worker_file_path)
                batch_buffer = [] 
            
            # Sleep logic
            if config.SCRAPING_DETAILS_CONFIG["stagger_mode"] == "random":
                delay = random.uniform(
                    config.SCRAPING_DETAILS_CONFIG["stagger_step_sec"],
                    config.SCRAPING_DETAILS_CONFIG["stagger_max_sec"],
                )
                time.sleep(delay)
        
        # Flush remaining items after loop finishes
        if batch_buffer:
            save_batch(batch_buffer, worker_file_path)

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        # Try to save whatever is in buffer even if error occurs
        if batch_buffer:
             save_batch(batch_buffer, worker_file_path)
    finally:
        driver.quit()
        
    return str(worker_file_path)

def scrape_urls_worker(worker_id: int, search_page_url: str, start_page: int, end_page: int, stop_event: threading.Event) -> list[str]:
    """
    Worker to scrape a range of pagination pages.
    """
    # Stagger start to prevent all browsers opening at exact same millisecond
    time.sleep(worker_id * 2.0)
    
    print(f"[Worker {worker_id}] Starting URL scrape for pages {start_page} to {end_page}...")
    
    driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
    scraper = Scraper(driver)
    found_urls = []

    try:
        found_urls = scraper.scrape_listing_urls(search_page_url, start_page, end_page)
        print(f"[Worker {worker_id}] Finished. Found {len(found_urls)} URLs.")
        
    except Exception as e:
        print(f"[Worker {worker_id}] Critical Error: {e}")
    finally:
        driver.quit()

    return found_urls