import os
import csv
import pandas as pd
from . import config


def save_urls_to_csv(urls, file_path):
    """Saves a list of URLs to a CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        writer.writerows([[u] for u in urls])

    print(f"Saved {len(urls)} URLs → {file_path}")


def save_details_to_csv(details, file_path):
    """
    Saves scraped details to a CSV file, supporting both overwrite and append modes.
    The behavior is controlled by the `append_mode` setting in `config.py`.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not details:
        print("No new details to save.")
        return

    file_exists = os.path.exists(file_path)
    is_append_mode = config.SCRAPING_DETAILS_CONFIG.get("append_mode", False)
    mode = 'a' if is_append_mode and file_exists else 'w'
    write_header = not (is_append_mode and file_exists)

    df = pd.DataFrame(details)
    df.to_csv(
        file_path,
        mode=mode,
        header=write_header,
        index=False,
        quoting=csv.QUOTE_ALL
    )

    if mode == 'a':
        print(f"Appended {len(details)} new listing records → {file_path}")
    else:
        print(f"Saved {len(details)} listing records → {file_path}")


def chunks(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    iterable = list(iterable)
    k, m = divmod(len(iterable), n)
    start = 0

    for i in range(n):
        end = start + k + (1 if i < m else 0)
        yield iterable[start:end]
        start = end