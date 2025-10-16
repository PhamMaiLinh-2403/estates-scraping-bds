import time 
import random
import threading
from seleniumbase import Driver

from . import config
from .scraping import Scraper


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

def scrape_worker(worker_id: int, url_subset: list[str], existing_ids: set[str], stop_event: threading.Event) -> list[dict]:
    """
    Defines the task for a single scraping worker.
    Each worker gets its own driver & scraper instance.
    """
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
            listing_id = str(data.get("id")).replace(".0", "") 
            if listing_id in existing_ids:
                print(f"[Worker {worker_id}] Skipping already-scraped ID: {listing_id}")
                continue
            results.append(data)

        if config.SCRAPING_DETAILS_CONFIG["stagger_mode"] == "random":
            delay = random.uniform(
                config.SCRAPING_DETAILS_CONFIG["stagger_step_sec"],
                config.SCRAPING_DETAILS_CONFIG["stagger_max_sec"],
            )
            time.sleep(delay)
    driver.quit()
    return results