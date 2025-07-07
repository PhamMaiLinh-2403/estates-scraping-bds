import json
import time
import re
from functools import wraps

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from . import config


def retry(max_tries=3, delay_seconds=3, backoff=2):
    """
    A decorator for retrying a function or method if it raises an exception.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            func_name = f"{args[0].__class__.__name__}.{func.__name__}"

            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1

                    if tries >= max_tries:
                        print(f"'{func_name}' failed after {max_tries} attempts. Last error: {e}")
                        return None

                    sleep_time = delay_seconds * (backoff ** (tries - 1))
                    print(f"'{func_name}' failed. Retrying in {sleep_time:.2f} seconds... ({tries}/{max_tries})")
                    time.sleep(sleep_time)

            return None
        return wrapper
    return decorator


class Scraper:
    """
    Scraping data utility functions, using a Selenium WebDriver instance.
    Optimized for robustness with explicit waits and retries.
    """
    def __init__(self, driver: WebDriver):
        """Initializes the scraper with a Selenium WebDriver instance."""
        self.driver = driver

    def scrape_listing_urls(self, search_page_url: str) -> list[str]:
        """
        Scrapes all listing URLs from a search results page by iterating through page numbers in the URL.
        """
        urls = []
        page_number = 1

        while True:
            # Construct URL for the current page
            if page_number == 1:
                current_url = search_page_url
            else:
                base_search_url = search_page_url.rstrip('/')
                current_url = f"{base_search_url}/p{page_number}"

            print(f"Scraping page {page_number}: {current_url}")
            self.driver.get(current_url)

            try:
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a.js__product-link-for-product-id"))
                )
            except TimeoutException:
                print(f"No product links found on page {page_number}. Assuming end of results.")
                break

            urls_before_scrape = len(urls)
            links = self.driver.find_elements(By.CSS_SELECTOR, "a.js__product-link-for-product-id")

            for link in links:
                href = link.get_attribute("href")

                if href:
                    full_url = config.BASE_URL + href if href.startswith("/") else href
                    if full_url not in urls:
                        urls.append(full_url)

            print(f"Collected {len(urls)} unique URLs so far...")

            if len(urls) == urls_before_scrape:
                print("No new URLs found on this page. Reached end of pagination.")
                break

            page_number += 1

        return urls

    @retry(max_tries=3, delay_seconds=5)
    def scrape_listing_details(self, url: str) -> dict | None:
        """
        Scrapes detailed information from a single property listing page.
        """
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, 'product-detail-web'))
            )
            body = self.driver.find_element(By.ID, 'product-detail-web')

            # Scrape coordinates from script tags
            coordinates = self._scrape_lat_long()

            listing_data = {
                "url": url,
                "title": self._get_text(self.driver, "h1.re__pr-title"),
                "latitude": coordinates.get("latitude"),
                "longitude": coordinates.get("longitude"),
                "short_address": self._get_text(self.driver, '.re__pr-short-description'),
                "address_parts": self._scrape_address_parts(),
                "main_info": self._scrape_info_items(body),
                "description": self._get_text(body, '.re__detail-content'),
                "other_info": self._scrape_other_info(body),
                "image_urls": self._scrape_og_images(),
            }
            return listing_data
        except (TimeoutException, Exception) as e:
            print(f"Attempt failed for {url}: {e}")
            raise e


    def _scrape_lat_long(self) -> dict:
        """
        Parse script tags to find and extract latitude and longitude using regex.
        """
        latitude, longitude = None, None
        try:
            script_tags = self.driver.find_elements(By.TAG_NAME, 'script')
            target_text = "initListingHistoryLazy"

            for script in script_tags:
                script_content = script.get_attribute('innerHTML')
                if script_content and target_text in script_content:
                    lat_match = re.search(r"latitude:\s*([\d\.]+)", script_content)
                    lon_match = re.search(r"longitude:\s*([\d\.]+)", script_content)

                    if lat_match:
                        latitude = float(lat_match.group(1))
                    if lon_match:
                        longitude = float(lon_match.group(1))
                    if latitude is not None and longitude is not None:
                        break
        except Exception as e:
            print(f"Could not parse lat/long from script tags: {e}")

        return {"latitude": latitude, "longitude": longitude}

    @staticmethod
    def _get_text(element, selector, by=By.CSS_SELECTOR):
        """
        Safely gets text from an element.
        """
        try:
            return element.find_element(by, selector).text.strip()
        except NoSuchElementException:
            return None

    def _scrape_info_items(self, body):
        items_data = []
        info_items = body.find_elements(By.CSS_SELECTOR, ".re__pr-short-info-item")

        for item in info_items:
            title = self._get_text(item, ".title")
            value = self._get_text(item, ".value")
            ext = self._get_text(item, ".ext")
            items_data.append({"title": title, "value": value, "ext": ext})
        return items_data

    def _scrape_other_info(self, body):
        other_info_dict = {}
        other_info_items = body.find_elements(By.CSS_SELECTOR, '.re__pr-other-info-display .re__pr-specs-content-item')

        for item in other_info_items:
            key = self._get_text(item, '.re__pr-specs-content-item-title')
            value = self._get_text(item, '.re__pr-specs-content-item-value')

            if key and value:
                other_info_dict[key] = value
        return other_info_dict

    def _scrape_og_images(self):
        meta_tags = self.driver.find_elements(By.CSS_SELECTOR, 'meta[property="og:image"]')
        return [tag.get_attribute("content") for tag in meta_tags if tag.get_attribute("content")]

    def _scrape_address_parts(self):
        try:
            # Primary method: Scrape breadcrumbs
            breadcrumb_items = self.driver.find_elements(By.CSS_SELECTOR, '.re__breadcrumb.js__breadcrumb .re__link-se')
            if breadcrumb_items:
                return [item.text.strip() for item in breadcrumb_items if item.text.strip()]
        except NoSuchElementException:
            pass

        # Fallback method: Scrape JSON-LD schema
        script_tags = self.driver.find_elements(By.CSS_SELECTOR, 'script[type="application/ld+json"]')

        for script_tag in script_tags:
            try:
                json_text = script_tag.get_attribute('innerHTML')
                data = json.loads(json_text)

                if data.get('@type') == 'BreadcrumbList':
                    return [item['name'] for item in data.get('itemListElement', []) if 'name' in item]
            except (json.JSONDecodeError, TypeError):
                continue
        return []