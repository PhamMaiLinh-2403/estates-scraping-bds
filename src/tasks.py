import time
import random
from .selenium_manager import create_stealth_driver
from .scraping import Scraper
from . import config


def scrape_worker(worker_id: int, url_subset: list[str]) -> list[dict]:
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
            results.append(data)
        if config.SCRAPING_DETAILS_CONFIG["stagger_mode"] == "random":
            delay = random.uniform(
                config.SCRAPING_DETAILS_CONFIG["stagger_step_sec"],
                config.SCRAPING_DETAILS_CONFIG["stagger_max_sec"],
            )
            time.sleep(delay)
    driver.quit()
    return results