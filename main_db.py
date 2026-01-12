#!/usr/bin/env python3
"""
Main script for real estate scraping with database storage.
Supports URL collection and detail scraping with deduplication.
"""

import threading
import signal
import sys
from pathlib import Path

from src import config
from src.database_manager import DatabaseManager
from src.selenium_manager import scrape_urls_worker, scrape_worker
from src.utils import split_page_ranges, chunks

# Global stop event for graceful shutdown
stop_event = threading.Event()
interrupt_count = 0

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global interrupt_count
    interrupt_count += 1
    
    if interrupt_count == 1:
        print("\n\n⚠️  Interrupt received. Stopping workers gracefully...")
        print("⏳ Please wait for current tasks to finish...")
        print("💡 Press Ctrl+C again to force quit (may lose data)")
        stop_event.set()
    else:
        print("\n\n❌ Force quit! Some data may be lost.")
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def get_config_snapshot():
    """Extract serializable configuration values."""
    return {
        'BASE_URL': config.BASE_URL,
        'SEARCH_PAGE_URL': config.SEARCH_PAGE_URL,
        'START_PAGE_NUMBER': config.START_PAGE_NUMBER,
        'END_PAGE_NUMBER': config.END_PAGE_NUMBER,
        'MAX_WORKERS': config.MAX_WORKERS,
        'SCRAPING_DETAILS_CONFIG': config.SCRAPING_DETAILS_CONFIG,
        'SELENIUM_CONFIG': config.SELENIUM_CONFIG,
    }

def scrape_urls_multithreaded(db_manager: DatabaseManager):
    """
    Phase 1: Scrape listing URLs from search pages using multiple workers.
    """
    global interrupt_count
    interrupt_count = 0  # Reset interrupt counter
    stop_event.clear()   # Clear stop event
    
    print("=" * 80)
    print("📋 PHASE 1: Scraping Listing URLs")
    print("=" * 80)
    
    # Start scraping session
    session_id = db_manager.start_scraping_session(
        scrape_type='URL_COLLECTION',
        config_snapshot=get_config_snapshot()
    )
    
    # Split page ranges among workers
    page_ranges = split_page_ranges(
        config.START_PAGE_NUMBER,
        config.END_PAGE_NUMBER,
        config.MAX_WORKERS
    )
    
    print(f"🔧 Starting {len(page_ranges)} workers for pages {config.START_PAGE_NUMBER}-{config.END_PAGE_NUMBER}")
    print(f"💡 Press Ctrl+C once to stop gracefully, twice to force quit\n")
    
    threads = []
    all_urls = []
    urls_lock = threading.Lock()
    
    def worker_wrapper(wid, sp, ep):
        """Wrapper to safely collect URLs from workers."""
        try:
            urls = scrape_urls_worker(wid, config.SEARCH_PAGE_URL, sp, ep, db_manager, stop_event)
            with urls_lock:
                all_urls.extend(urls)
        except Exception as e:
            print(f"[Worker {wid}] URL scraping exception: {e}")
    
    for worker_id, (start_page, end_page) in enumerate(page_ranges):
        thread = threading.Thread(
            target=worker_wrapper,
            args=(worker_id, start_page, end_page),
            daemon=False
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete with periodic checks
    try:
        while any(t.is_alive() for t in threads):
            for thread in threads:
                thread.join(timeout=0.5)
            
            if stop_event.is_set():
                print("\n⏳ Waiting for workers to finish current page (max 30s)...")
                for thread in threads:
                    thread.join(timeout=30)
                break
    except KeyboardInterrupt:
        print("\n⚠️  Main thread interrupted, signaling workers...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=30)
    
    if stop_event.is_set():
        print("\n⚠️  URL collection interrupted by user")
    
    # Update session statistics
    stats = {
        'total_urls': len(all_urls),
        'successful_scrapes': len(all_urls),
        'failed_scrapes': 0,
        'status': 'COMPLETED' if not stop_event.is_set() else 'INTERRUPTED'
    }
    
    db_manager.end_scraping_session(session_id, stats)
    
    print(f"\n✅ URL collection complete. Found {len(all_urls)} URLs")
    print(f"📊 Session ID: {session_id}")
    
    return all_urls

def scrape_details_multithreaded(db_manager: DatabaseManager, urls: list[str] = None):
    """
    Phase 2: Scrape detailed information for each listing using multiple workers.
    If urls not provided, fetches pending URLs from database.
    """
    global interrupt_count
    interrupt_count = 0  # Reset interrupt counter
    stop_event.clear()   # Clear stop event
    
    print("\n" + "=" * 80)
    print("🏠 PHASE 2: Scraping Listing Details")
    print("=" * 80)
    
    # Get URLs to scrape
    if urls is None:
        urls = db_manager.get_pending_urls()
        print(f"📥 Fetched {len(urls)} pending URLs from database")
    
    if not urls:
        print("⚠️  No URLs to scrape")
        return
    
    # Apply config limits
    start_idx = config.SCRAPING_DETAILS_CONFIG.get("start_index", 0)
    count = config.SCRAPING_DETAILS_CONFIG.get("count", 0)
    
    if count > 0:
        urls = urls[start_idx:start_idx + count]
        print(f"🎯 Processing {len(urls)} URLs (from index {start_idx})")
    else:
        urls = urls[start_idx:]
        print(f"🎯 Processing {len(urls)} URLs (from index {start_idx} to end)")
    
    # Start scraping session
    session_id = db_manager.start_scraping_session(
        scrape_type='DETAIL_SCRAPING',
        config_snapshot=get_config_snapshot()
    )
    
    # Split URLs among workers
    url_chunks = list(chunks(urls, config.MAX_WORKERS))
    print(f"🔧 Starting {len(url_chunks)} workers")
    print(f"💡 Press Ctrl+C once to stop gracefully, twice to force quit\n")
    
    threads = []
    worker_stats = []
    
    # Use a lock to safely append stats
    stats_lock = threading.Lock()
    
    def worker_wrapper(wid, subset):
        """Wrapper to safely collect stats from workers."""
        try:
            result = scrape_worker(wid, subset, db_manager, stop_event, session_id)
            with stats_lock:
                worker_stats.append(result)
        except Exception as e:
            print(f"[Worker {wid}] Wrapper exception: {e}")
            with stats_lock:
                worker_stats.append({
                    'total_urls': len(subset),
                    'successful_scrapes': 0,
                    'failed_scrapes': len(subset),
                    'new_records': 0,
                    'changed_records': 0,
                    'duplicate_records': 0,
                    'status': 'FAILED',
                    'error_message': str(e)
                })
    
    for worker_id, url_subset in enumerate(url_chunks):
        thread = threading.Thread(
            target=worker_wrapper,
            args=(worker_id, url_subset),
            daemon=False  # Don't make daemon so we can join properly
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all workers to complete with periodic checks
    try:
        while any(t.is_alive() for t in threads):
            for thread in threads:
                thread.join(timeout=0.5)  # Check every 0.5 seconds
            
            # If interrupted, give threads time to finish current task
            if stop_event.is_set():
                print("\n⏳ Waiting for workers to finish current tasks (max 30s)...")
                for thread in threads:
                    thread.join(timeout=30)
                break
    except KeyboardInterrupt:
        # This catches Ctrl+C in the main thread
        print("\n⚠️  Main thread interrupted, signaling workers...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=30)
    
    if stop_event.is_set():
        print("\n⚠️  Detail scraping interrupted by user")
    
    # Aggregate statistics
    total_stats = {
        'total_urls': sum(s.get('total_urls', 0) for s in worker_stats),
        'successful_scrapes': sum(s.get('successful_scrapes', 0) for s in worker_stats),
        'failed_scrapes': sum(s.get('failed_scrapes', 0) for s in worker_stats),
        'new_records': sum(s.get('new_records', 0) for s in worker_stats),
        'changed_records': sum(s.get('changed_records', 0) for s in worker_stats),
        'duplicate_records': sum(s.get('duplicate_records', 0) for s in worker_stats),
        'status': 'COMPLETED' if not stop_event.is_set() else 'INTERRUPTED'
    }
    
    # Update session
    db_manager.end_scraping_session(session_id, total_stats)
    
    # Print summary
    print("\n" + "=" * 80)
    if stop_event.is_set():
        print("⚠️  SCRAPING INTERRUPTED BY USER")
    else:
        print("📊 SCRAPING SUMMARY")
    print("=" * 80)
    print(f"Total URLs:        {total_stats['total_urls']}")
    print(f"Successful:        {total_stats['successful_scrapes']}")
    print(f"Failed:            {total_stats['failed_scrapes']}")
    print(f"New Records:       {total_stats['new_records']}")
    print(f"Changed Records:   {total_stats['changed_records']}")
    print(f"Duplicates:        {total_stats['duplicate_records']}")
    print(f"Status:            {total_stats['status']}")
    print(f"Session ID:        {session_id}")
    print("=" * 80)

def export_data(db_manager: DatabaseManager, output_dir: Path = None):
    """Export database tables to Excel files."""
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📤 EXPORTING DATA")
    print("=" * 80)
    
    # Export raw listings
    raw_output = output_dir / "raw_listings.xlsx"
    db_manager.export_to_excel('raw_listings', str(raw_output))
    
    # Export cleaned listings
    cleaned_output = output_dir / "cleaned_listings.xlsx"
    db_manager.export_to_excel('cleaned_listings', str(cleaned_output))
    
    # Export metadata
    metadata_output = output_dir / "scraping_metadata.xlsx"
    db_manager.export_to_excel('scraping_metadata', str(metadata_output))
    
    print("=" * 80)

def show_statistics(db_manager: DatabaseManager):
    """Display database statistics."""
    print("\n" + "=" * 80)
    print("📈 DATABASE STATISTICS")
    print("=" * 80)
    
    stats = db_manager.get_statistics()
    
    # Raw listings
    raw = stats['raw_listings']
    print(f"\nRaw Listings:")
    print(f"  Total:       {raw['total']}")
    print(f"  New:         {raw['new']}")
    print(f"  Changed:     {raw['changed']}")
    print(f"  Duplicates:  {raw['duplicate']}")
    
    # Cleaned listings
    cleaned = stats['cleaned_listings']
    print(f"\nCleaned Listings:")
    print(f"  Total:       {cleaned['total']}")
    
    # URL queue
    queue = stats.get('url_queue', {})
    if queue:
        print(f"\nURL Queue:")
        for status, count in queue.items():
            print(f"  {status:12} {count}")
    
    # Recent sessions
    recent = stats.get('recent_sessions', [])
    if recent:
        print(f"\nRecent Sessions (last 5):")
        for session in recent[:5]:
            print(f"  [{session['id']}] {session['scrape_type']:20} "
                  f"{session['status']:12} {session['start_time']}")
    
    print("=" * 80)

def main():
    """Main entry point."""
    # Initialize database
    db_manager = DatabaseManager(db_path=str(config.OUTPUT_DIR / "real_estate.db"))
    
    print("\n🚀 Real Estate Scraping Pipeline (Database Version)")
    print("=" * 80)
    
    # Show current statistics
    show_statistics(db_manager)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Scrape URLs only")
    print("2. Scrape details only (from pending URLs)")
    print("3. Full pipeline (URLs + Details)")
    print("4. Export data to Excel")
    print("5. Show statistics")
    print("6. Exit")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        scrape_urls_multithreaded(db_manager)
    elif choice == '2':
        scrape_details_multithreaded(db_manager)
    elif choice == '3':
        urls = scrape_urls_multithreaded(db_manager)
        if urls and not stop_event.is_set():
            scrape_details_multithreaded(db_manager, urls)
    elif choice == '4':
        export_data(db_manager)
    elif choice == '5':
        show_statistics(db_manager)
    elif choice == '6':
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice")
        return
    
    # Show final statistics
    show_statistics(db_manager)
    
    # Ask if user wants to export
    export_choice = input("\nExport data to Excel? (y/n): ").strip().lower()
    if export_choice == 'y':
        export_data(db_manager)
    
    print("\n✅ All done!")

if __name__ == "__main__":
    main()