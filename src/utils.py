import csv
import pandas as pd
from pathlib import Path
from typing import List, Any, Union
from src import config

def ensure_dir(file_path: Union[str, Path]):
    """Ensures the directory for the given file path exists."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def save_to_csv(data: List[Any], file_path: Union[str, Path], is_url_list: bool = False):
    """
    Universal CSV saver. Handles URL lists and detailed dictionaries.
    Supports append/overwrite logic based on config.
    """
    path = Path(file_path)
    ensure_dir(path)

    if not data:
        print(f"No data to save to {path.name}.")
        return

    # Determine save mode
    file_exists = path.exists()
    is_append = config.SCRAPING_DETAILS_CONFIG.get("append_mode", False)
    # URLs are usually overwritten, details are usually appended
    mode = 'a' if is_append and file_exists and not is_url_list else 'w'
    write_header = not (is_append and file_exists) or is_url_list

    if is_url_list:
        # Simple list of URLs
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["url"])
            writer.writerows([[u] for u in data])
    else:
        # List of dictionaries (listing details)
        df = pd.DataFrame(data)
        df.to_csv(
            path,
            mode=mode,
            header=write_header,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding='utf-8'
        )

    print(f"{'Appended' if mode == 'a' else 'Saved'} {len(data)} records to {path}")

def chunks(iterable, n):
    """Split iterable into n roughly equal chunks."""
    lst = list(iterable)
    k, m = divmod(len(lst), n)
    for i in range(n):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]

def build_page_url(search_page_url, page_number):
    if page_number == 1:
        return search_page_url
    base_search_url = search_page_url.rstrip('/')
    return f"{base_search_url}/p{page_number}"

def split_page_ranges(start, end, n_workers):
    """Splits a range of pages (e.g. 1-100) into n chunks."""
    total_pages = end - start + 1
    if n_workers <= 0 or total_pages <= 0:
        return []

    # If fewer pages than workers, reduce workers
    n_workers = min(n_workers, total_pages)
    
    chunk_size = total_pages // n_workers
    ranges = []
    current_start = start

    for i in range(n_workers):
        current_end = current_start + chunk_size - 1
        # Add remainder to the last worker
        if i == n_workers - 1:
            current_end = end
        
        ranges.append((current_start, current_end))
        current_start = current_end + 1
    
    return ranges

# --- Backward compatibility aliases ---
def save_urls_to_csv(urls, file_path):
    save_to_csv(urls, file_path, is_url_list=True)

def save_details_to_csv(details, file_path):
    save_to_csv(details, file_path, is_url_list=False)