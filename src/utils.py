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
    """Yield successive n-sized chunks from an iterable."""
    lst = list(iterable)
    k, m = divmod(len(lst), n)
    for i in range(n):
        start = i * k + min(i, m)
        end = (i + 1) * k + min(i + 1, m)
        yield lst[start:end]

# --- Backward compatibility aliases ---
def save_urls_to_csv(urls, file_path):
    save_to_csv(urls, file_path, is_url_list=True)

def save_details_to_csv(details, file_path):
    save_to_csv(details, file_path, is_url_list=False)