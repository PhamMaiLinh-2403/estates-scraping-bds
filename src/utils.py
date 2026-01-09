import csv
import pandas as pd
import shutil
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

def save_batch(data: List[dict], file_path: Path):
    """
    Appends a batch of data to a specific temp file.
    Creates the file and writes the header if it doesn't exist.
    """
    if not data:
        return

    file_exists = file_path.exists()
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(
            file_path,
            mode='a',
            header=not file_exists, # Write header only if file didn't exist
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding='utf-8'
        )
    except Exception as e:
        print(f"Error saving batch to {file_path}: {e}")

def merge_temp_files(temp_dir: Path, output_file: Path, append_mode: bool = True):
    """
    Consolidates all CSV files in temp_dir into output_file and deletes temp_dir.
    """
    temp_files = list(temp_dir.glob("*.csv"))
    if not temp_files:
        print("No temp files to merge.")
        return

    print(f"Merging {len(temp_files)} temp files into {output_file}...")
    
    ensure_dir(output_file)
    
    # Check if target exists to decide on header
    target_exists = output_file.exists() and append_mode
    write_header = not target_exists
    
    total_records = 0
    
    try:
        # Iterate over temp files and append them to the main file
        for i, temp_file in enumerate(temp_files):
            try:
                df_chunk = pd.read_csv(temp_file, on_bad_lines='skip')
                
                if not df_chunk.empty:
                    df_chunk.to_csv(
                        output_file,
                        mode='a',
                        header=write_header,
                        index=False,
                        quoting=csv.QUOTE_ALL,
                        encoding='utf-8-sig'
                    )
                    total_records += len(df_chunk)
                    write_header = False # Only write headers on the first time 
            except Exception as e:
                print(f"Failed to merge temp file {temp_file}: {e}")
        
        print(f"Successfully merged {total_records} records.")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temp directory: {temp_dir}")
        
    except Exception as e:
        print(f"Critical error during merge: {e}")

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