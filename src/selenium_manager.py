import time 
import random
import threading
from seleniumbase import Driver

from . import config
from .scraping import Scraper
from .database_manager import DatabaseManager

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

def scrape_worker(worker_id: int, url_subset: list[str], db_manager: DatabaseManager, 
                 stop_event: threading.Event, session_id: int) -> dict:
    """
    Worker to scrape listing details and store them in database.
    Returns statistics about the scraping session.
    """
    base = config.SCRAPING_DETAILS_CONFIG.get("stagger_step_sec", 2.0)
    start_delay = worker_id * base
    print(f"[Worker {worker_id}]: Sleeping {start_delay:.1f}s before start.")
    time.sleep(start_delay)

    driver = None
    stats = {
        'total_urls': len(url_subset),
        'successful_scrapes': 0,
        'failed_scrapes': 0,
        'new_records': 0,
        'changed_records': 0,
        'duplicate_records': 0,
        'status': 'COMPLETED'
    }

    try:
        driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
        scraper = Scraper(driver)
        
        for idx, url in enumerate(url_subset, 1):
            # Check stop event before processing
            if stop_event.is_set():
                stats['status'] = 'INTERRUPTED'
                print(f"[Worker {worker_id}] ⚠️  Stopping due to interrupt signal")
                break

            print(f"[Worker {worker_id}]  {idx}/{len(url_subset)}  → {url}")
            
            try:
                # Scrape the listing
                data = scraper.scrape_listing_details(url)

                if data:
                    # Insert into database IMMEDIATELY
                    record_id, status = db_manager.insert_raw_listing(data)
                    
                    # Update statistics
                    stats['successful_scrapes'] += 1
                    if status == 'NEW':
                        stats['new_records'] += 1
                    elif status == 'CHANGED':
                        stats['changed_records'] += 1
                    else:  # DUPLICATE
                        stats['duplicate_records'] += 1
                    
                    # Update URL queue status
                    db_manager.update_url_status(url, 'COMPLETED')
                    
                    print(f"[Worker {worker_id}] ✅ Saved listing {data.get('id')} as {status} (ID: {record_id})")
                else:
                    stats['failed_scrapes'] += 1
                    db_manager.update_url_status(url, 'FAILED', 'No data returned')
                    
            except Exception as e:
                stats['failed_scrapes'] += 1
                error_msg = str(e)
                print(f"[Worker {worker_id}] ❌ Error scraping {url}: {error_msg}")
                try:
                    db_manager.update_url_status(url, 'FAILED', error_msg)
                except:
                    pass  # Don't fail on database errors during error handling
            
            # Check stop event before sleeping
            if stop_event.is_set():
                stats['status'] = 'INTERRUPTED'
                print(f"[Worker {worker_id}] ⚠️  Stopping due to interrupt signal")
                break
            
            # Sleep logic with interrupt checking
            if config.SCRAPING_DETAILS_CONFIG["stagger_mode"] == "random":
                delay = random.uniform(
                    config.SCRAPING_DETAILS_CONFIG["stagger_step_sec"],
                    config.SCRAPING_DETAILS_CONFIG["stagger_max_sec"],
                )
                # Sleep in small chunks to respond quickly to interrupts
                elapsed = 0
                while elapsed < delay and not stop_event.is_set():
                    chunk = min(0.2, delay - elapsed)
                    time.sleep(chunk)
                    elapsed += chunk

    except KeyboardInterrupt:
        print(f"\n[Worker {worker_id}] ⚠️  Keyboard interrupt received")
        stats['status'] = 'INTERRUPTED'
    except Exception as e:
        stats['status'] = 'FAILED'
        stats['error_message'] = str(e)
        print(f"[Worker {worker_id}] ❌ Critical Error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass  # Ignore errors during cleanup
        
        if stats['successful_scrapes'] > 0:
            print(f"[Worker {worker_id}] 📊 Completed {stats['successful_scrapes']}/{stats['total_urls']} before stopping")
        
    return stats

def scrape_urls_worker(worker_id: int, search_page_url: str, start_page: int, 
                       end_page: int, db_manager: DatabaseManager, 
                       stop_event: threading.Event) -> list[str]:
    """
    Worker to scrape a range of pagination pages and store URLs in database.
    """
    # Stagger start to prevent all browsers opening at exact same millisecond
    time.sleep(worker_id * 2.0)
    
    print(f"[Worker {worker_id}] Starting URL scrape for pages {start_page} to {end_page}...")
    
    driver = None
    found_urls = []

    try:
        driver = create_stealth_driver(headless=config.SELENIUM_CONFIG["headless"])
        scraper = Scraper(driver)
        
        for page_num in range(start_page, end_page + 1):
            # Check for interrupt before each page
            if stop_event.is_set():
                print(f"[Worker {worker_id}] ⚠️  Stopping URL scraping due to interrupt")
                break
            
            page_urls = scraper.scrape_listing_urls(search_page_url, page_num, page_num)
            
            if page_urls:
                # Store URLs in database immediately
                db_manager.add_urls_to_queue(page_urls)
                found_urls.extend(page_urls)
                print(f"[Worker {worker_id}] ✅ Page {page_num}: Added {len(page_urls)} URLs to queue")
            else:
                print(f"[Worker {worker_id}] ⚠️  Page {page_num}: No URLs found")
            
            # Check for interrupt before sleeping
            if stop_event.is_set():
                break
            
            # Small delay between pages
            if page_num < end_page:
                elapsed = 0
                delay = 1.0
                while elapsed < delay and not stop_event.is_set():
                    chunk = min(0.2, delay - elapsed)
                    time.sleep(chunk)
                    elapsed += chunk
        
        print(f"[Worker {worker_id}] Finished. Found {len(found_urls)} URLs total.")
        
    except KeyboardInterrupt:
        print(f"[Worker {worker_id}] ⚠️  Keyboard interrupt received")
    except Exception as e:
        print(f"[Worker {worker_id}] ❌ Critical Error: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

    return found_urls