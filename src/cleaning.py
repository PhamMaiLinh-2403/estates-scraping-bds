import re
import json
import time 
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import random
from rapidfuzz import fuzz, process
from math import * 
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

import osmnx as ox
from shapely.geometry import Point
from shapely.ops import nearest_points
from geopy.distance import geodesic

from src.config import *

class DataCleaner:
    """
    Static methods for cleaning and standardizing scraped data.
    """

    # -- Helper methods --
    @staticmethod
    def _parse_and_clean_number(text_value):
        """
        Extract the first numeric token from a string and return it as a float.
        """
        if not isinstance(text_value, str):
            return None

        match = re.search(r"([\d\.,]+)", text_value)
        if not match:
            return None

        num_str = match.group(1)

        # 1. Remove the thousand separators ('.').
        # 2. Replace the decimal separator (',') with a standard period ('.').
        cleaned_num_str = num_str.replace(".", "").replace(",", ".")

        try:
            return round(float(cleaned_num_str), 2)
        except (ValueError, TypeError):
            return None
        
    @staticmethod     
    def _is_negated(text, keyword):
        negation_keywords = ['không', 'chưa', 'ko', "đường", "ngõ", "hẻm", "ngách", "kiệt"]
        window = 4  

        for keyword in negation_keywords:
            pattern = rf'{keyword}(?:\s+\w+){{0,{window}}}?\s+{re.escape(keyword)}'
            if re.search(pattern, text):
                return True
        return False
    
    @staticmethod
    def clean_description_text(raw_text):
        """
        Cleans a real estate description by removing seller contact information.
        """
        if not isinstance(raw_text, str) or pd.isna(raw_text):
            return ""  
        
        # Pattern to detect the start of the seller info block
        contact_start_pattern = re.compile(
            r'(lh|liên hệ|zalo|sđt|phone|call|chuyên bđs|số điện thoại)',
            re.IGNORECASE
        )

        # Search for the first occurrence of seller contact indicators
        match = contact_start_pattern.search(raw_text.lower())
        
        if match:
            # Truncate the text from the match onward
            cleaned = raw_text[:match.start()]
        else:
            cleaned = raw_text

        cleaned = re.sub(r'\n{2,}', '\n\n', cleaned.strip())

        return cleaned
    
    @staticmethod
    def _is_on_main_road(text: str, lat, lon):
        text = DataCleaner.clean_description_text(text.lower().strip())

        geolocator = Nominatim(user_agent="vn_address_distance")

        not_on_main_road = r"mặt ngõ|mặt hẻm|gần phố|sát phố|nhà hẻm|nhà ngõ|ngõ vào|ngõ thông|hẻm vào|hẻm thông|ngõ|hẻm|kiệt|ngách"
        major_roads = 'đường|phố|mặt đường|mặt phố|mp|vành đai|đại lộ|đl|mặt tiền|mt|trục chính|quốc lộ|ql|qlo|tỉnh lộ|tl|ngã tư|ngã ba|ngã 4|ngã 3'
        on_main_road = "\b(?:ngay|trên|nằm)?\s*(?:mặt tiền|mặt phố|mặt đường|phố|đường)\b"
        distance_to_main_road = r"(?:vài\s*bước|[0-9]+\s*m|[0-9]+m|mấy\s*m|chỉ\s*\d+\s*m)(?:\s+\w+){0,3}\s+(?:ra|tới)\s+(?:mặt\s*tiền|đường|phố|đường\s+chính|mặt\s*phố)"

        if re.search(not_on_main_road, text):
            return "Mặt ngõ"
        elif re.search(rf'(?:cách|ra|gần|sát)(?:\s+\w+){{0,5}}\s+(?:{major_roads})', text, re.IGNORECASE):
            return "Mặt ngõ"
        elif re.search(distance_to_main_road, text, re.IGNORECASE):
            return "Mặt ngõ"
        elif lat is not None and lon is not None:
            lat, lon = float(lat), float(lon)
            try:
                location = geolocator.reverse((lat, lon), language="vi", timeout=10)
                print(f"Successfully completed reverse geocoding for coordinates: ({lat}, {lon})")
                time.sleep(random.uniform(1, 2))  

                if location and location.address:
                    address = location.address.lower()
                    for loc in ["ngõ", "hẻm", "ngách"]:
                        if loc in address:
                            return "Mặt ngõ"
                
                if re.search(on_main_road, text) or "nhà phố" in text.lower():
                    return "Mặt phố"

            except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
                print(f"Reverse geocoding request failed for ({lat}, {lon}): {e}")
                time.sleep(1) # Wait a bit before the next request on failure

        return None

    # -- Static Cleaning Methods -- 
    @staticmethod
    def extract_city(row):
        try:
            address_list = json.loads(row["address_parts"])
            if len(address_list) > 1:
                return address_list[1]
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            pass

        short_address = row.get("short_address")
        if pd.notna(short_address):
            parts = [p.strip() for p in str(short_address).split(",")]
            if parts:
                return parts[-1]
        return None

    @staticmethod
    def extract_district(row):
        try:
            address_list = json.loads(row["address_parts"])
            if len(address_list) > 2:
                return address_list[2]
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            pass

        short_address = row.get("short_address")
        if pd.notna(short_address):
            parts = [p.strip() for p in str(short_address).split(",")]
            if len(parts) >= 2:
                return parts[-2]
        return None

    @staticmethod
    def extract_ward(row):
        """Extracts the ward (Phường/Xã/Thị trấn) from the address."""
        short_address = row.get("short_address")
        if pd.notna(short_address) and isinstance(short_address, str):
            parts = [p.strip() for p in short_address.split(",")]
            for part in parts:
                if part.lower().startswith(("phường ", "xã ", "thị trấn ")):
                    return part

        try:
            address_list = json.loads(row["address_parts"])
            if address_list:
                last_item = address_list[-1]
                match = re.search(r"tại\s+((?:phường|xã|thị trấn)\s+[\w\s\d\-()]+)", last_item, re.IGNORECASE)
                if match:
                    return match.group(1).title().strip()
        except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            pass

        return None
    
    @staticmethod
    def extract_street(row):
        """Extracts the street name from the address."""
        short_address = str(row.get("short_address", "")).strip()
        if short_address:
            parts = [p.strip() for p in short_address.split(",")]
            for part in parts:
                match = re.search(r'\b(đường|phố|quốc lộ|đại lộ|xa lộ)\s+[\w\s\-()\/]+', part.lower())
                if match and len(match.group(0).split()) <= 5:
                    return match.group(0).title().strip()

        parts_raw = str(row.get("address_parts", "")).strip()
        if parts_raw:
            try:
                addr_list = json.loads(parts_raw)
                if addr_list:
                    last_item = addr_list[-1]
                    m = re.search(r"tại (đường|phố|quốc lộ|đại lộ|xa lộ)\s+([^,]+)", last_item, re.IGNORECASE)
                    if m and len(m.group(0).split()) <= 5:
                        return f"{m.group(1).capitalize()} {m.group(2).strip()}"
            except (json.JSONDecodeError, ValueError, SyntaxError):
                pass

        title = str(row.get("title", "")).strip()
        if title:
            m = re.search(r"(đường|phố|quốc lộ|đại lộ|xa lộ)\s+([\w\s\d\-]+?)(?:,|$|\s-|\s--|\()", title, re.IGNORECASE)
            if m and len(m.group(0).split()) <= 5:
                street_cap = " ".join(w.capitalize() for w in m.group(2).strip().split())
                return f"{m.group(1).capitalize()} {street_cap}"

        return None
    
    @staticmethod
    def extract_address_details(row):
        """
        Extract detailed address information (house number, alley, etc.).
        """
        short_address = str(row.get("short_address", "")).strip()
        text = DataCleaner.clean_description_text(f"{row.get('title', '')} {row.get('description', '')}".lower().strip())
        parts = [p.strip() for p in short_address.split(",") if pd.notna(p) and isinstance(p, str)]
        lat, lon = row.get("latitude", ""), row.get("longitude", "")
        address_details = []
        ignore_prefixes = ['đường', 'phố', 'phường', 'quận', 'huyện']

        for part in parts:
            lower_part = part.lower().strip()

            # Skip if it starts with an ignore prefix followed by "số"
            if any(lower_part.startswith(f"{prefix} số") for prefix in ignore_prefixes):
                continue

            if re.search(r'\b(số|ngõ|hẻm|ngách|kiệt|tổ|khu phố|khu vực|khu đô thị|khu công nghiệp|kđt|khu dân cư|dự án)\b', lower_part):
                address_details.append(part)


        if address_details:
            return ", ".join(address_details)
        elif (
            len(parts) > 3
            and not parts[0].lower().startswith((
                "đường", "phố", "quốc lộ", "đại lộ", "xa lộ"
            ))
            and (
                re.search(r'\b(hẻm|ngõ|ngách|kiệt|dự án|khu đô thị|kđt|khu dân cư|ấp|tổ|khu phố)\b', parts[0])
                or re.search(r"\d+", parts[0])
            )
        ):
            return parts[0]
        return DataCleaner._is_on_main_road(text, lat, lon)
            
    @staticmethod
    def extract_total_area(row):
        if pd.notna(row.get("main_info")):
            try:
                for item in json.loads(row["main_info"]):
                    if item.get("title") == "Diện tích":
                        area = DataCleaner._parse_and_clean_number(item.get("value"))
                        if pd.notna(area):
                            return area
            except json.JSONDecodeError:
                pass

        if pd.notna(row.get("other_info")):
            try:
                area_val = json.loads(row["other_info"]).get("Diện tích")
                area = DataCleaner._parse_and_clean_number(area_val)
                if pd.notna(area):
                    return area
            except (json.JSONDecodeError, TypeError):
                pass

        return np.nan
    
    @staticmethod
    def extract_price(row):
        """
        Extract and clean the price from a row.
        """
        price_str = None

        if pd.notna(row.get("main_info")):
            try:
                info_items = json.loads(row["main_info"])

                if len(info_items) > 0:
                    for idx in [0, 1]:
                        item = info_items[idx]
                        title = item.get("title", "").strip().lower()
                        if "giá" in title:
                            price_str = item.get("value")
                            break
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

        if not price_str and pd.notna(row.get("other_info")): # Fallback to other_info if not found in main_info
            try:
                other_data = json.loads(row["other_info"])
                for key, value in other_data.items():
                    if isinstance(key, str) and "giá" in key.lower():
                        price_str = value
                        break
            except (json.JSONDecodeError, TypeError):
                pass

        if pd.isna(price_str) or not price_str or any(term in price_str.lower() for term in ["thỏa thuận", "liên hệ"]):
            return np.nan 

        # Extract numeric value in the price string
        match = re.search(r"([\d\.,]+)", price_str)
        if not match:
            return np.nan

        try:
            number = float(DataCleaner._parse_and_clean_number(match.group(1)))
        except ValueError:
            return np.nan

        # Extract unit price
        unit_match = re.search(r'\d[\d\.,]*\s*([^\d\s]+(?:/[^\d\s]+)?)', price_str)
        if not unit_match:
            return np.nan
        unit = unit_match.group(1)

        area_val = row.get("area")
        if pd.isna(area_val):
            area_val = DataCleaner.extract_total_area(row)

        # Price conversion logic
        if unit == "tỷ":
            value = number * 1e9
        elif unit == "triệu":
            value = number * 1e6
        elif unit == "nghìn":
            value = number * 1e3
        elif unit == "triệu/m²" and pd.notna(area_val):
            value = number * area_val * 1e6
        elif unit == "nghìn/m²" and pd.notna(area_val):
            value = number * area_val * 1e3
        else:
            value = None

        return round(value, 2) if value else np.nan

    
    @staticmethod
    def extract_published_date(main_info_json: str):
        try:
            for item in json.loads(main_info_json):
                if item.get("title") == "Ngày đăng":
                    return item.get("value")
        except (json.JSONDecodeError, TypeError):
            pass
        return None
    
    @staticmethod
    def extract_num_floors(row):
        cleaned_text = (str(row.get('title') or '') + DataCleaner.clean_description_text(str(row.get('description') or '')).lower().strip())

        # --- TH1: Extract from other_info ---
        other_info = row.get("other_info", "")
        if isinstance(other_info, str):
            try:
                other_info = json.loads(other_info)
                ele = other_info.get("Số tầng")
                if ele is not None:
                    return int(re.search(r"\d+", ele).group())
            except Exception:
                pass
        if isinstance(other_info, dict) and other_info.get("Số tầng"):
            return int(other_info["Số tầng"].split()[0])

        # --- TH2: Nhà cũ/nát hoặc nhà cấp 4 ---
        if re.search(r"nhà (?:\w+\s*){0,3}(?:cũ|nát)", cleaned_text):
            return 0
        if re.search(r'nhà cấp 4|nhà c4|cấp 4|nc4', cleaned_text):
            return 1
    
    @staticmethod
    def extract_facade_count(row):
        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Số mặt tiền")

            if val:
                m = re.search(r"\d+", str(val))
                if m:
                    return int(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower().strip()
        for pattern, val in FACADE_COUNT_MAP:
            if re.search(pattern, text):
                return val
        return 1

    @staticmethod
    def extract_land_shape(row):
        text = f"{row.get('title', '')} {row.get('description', '')}".lower().strip()
        text = DataCleaner.clean_description_text(text)

        for shape, kws in SHAPE_KEYWORDS.items():
            for kw in kws:
                if re.search(r'\b' + re.escape(kw) + r'\b', text):
                    if not DataCleaner._is_negated(text, kw):
                        return shape
        return "Chữ nhật"
    
    @staticmethod
    def estimate_remaining_quality(row):
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        text = DataCleaner.clean_description_text(text)
        result = {}
        default_quality = 0.75 

        for quality_val, keywords in QUALITY_LEVELS:
            for kw in keywords:
                pattern = kw.replace(' ', '(?:\s*\w+\s*){0,2} ')
                pattern = pattern.strip()
                pattern = '\W' + pattern + '\W'
                qual = re.search(pattern, text)
                if qual:
                    ratio = fuzz.ratio(kw, qual.group(0))
                    if quality_val == 0 or quality_val == 1:
                        ratio += 3
                    result[quality_val] = [qual.group(0), ratio]
        if result:
            # Sort by ratio first, then by quality value
            best_quality, (match, score) = max(result.items(), key=lambda x: (x[1][1], x[0]))
            return best_quality
        else:
            return default_quality
        
    @staticmethod
    def extract_construction_cost(row):
        def check_basement(text):
            string = re.findall(pattern=r'(?:(?!được xây|giấy phép xây dựng|gpxd|cải tạo)\b\w+\b\W+){1,7}hầm(?:(?!\schui)\W+\b\w+\b){1,7}', string=text)
            if string:
                for substr in string:
                    basement = re.search(pattern=r'tầng|lầu|tấm|mê|\d+|xe|kết cấu|kc|ô tô|thang máy|trệt|lửng', string=substr)
                    if basement:
                        return True
            return False
        
        text = f"{row.get('title', '')} {row.get('description', '')}".lower().strip()
        title = row.get("title").lower().strip()
        description = str(row.get("description") or "").lower().strip()

        if isinstance(text, str) and pd.notnull(text):
            text = text.replace(',',' ').replace('+',' ').replace('\n',' ').replace('*', ' ')
            text = ' '.join(text.split())

            if 'nhà trệt' in text:
                if not re.search(r'nhà trệt\s*(?:\S+\s+){0,2}(?:\d*\s*)(?:tầng|lầu|tấm|mê|lửng|gác)', text):
                    return 4_000_000
            if re.search(pattern = r'nhà cấp 4|nhà c4|cấp 4\W|nc4|nhà trệt|nhà nát', string=text):
                return 4_000_000
            if row['Số tầng công trình'] == 1:
                return 6_275_876
            
        if isinstance(title, str) and pd.notnull(title):
            if check_basement(title) and (re.search(r'biệt thự', title) or re.search(r'villa\W', title)):
                return 12_848_184
            return 10_510_920
        
        if isinstance(description, str) and not description.isnull():
            if re.search(r'biệt thự', description) or re.search(r'villa\W', description):
                villa_pattern = [
                    r'(?:thiết kế|xây)*\s*(?:\S+\s+){0,5} (?:phong cách|kiểu|dạng|kiến trúc|cấu trúc)\s*(?:\S+\s+){0,2} (?:biệt thự|villa\W)', # Các nhà có cấu trúc villa
                    r'bán (?:(?!mua|xây)\S+\s+){0,3}(?:biệt thự|villa\W)', # Bán biệt thự
                    r'(?:biệt thự|villa\W)\s*(?:\S+\s+){0,3}\d+\s*tầng' # Biệt thự bao nhiêu tầng
                ]
                not_villa_pattern = [
                    r'(?:đối diện|nằm|sát|cạnh|ngay|liền kề|hàng xóm|xung quanh|gần|view|nhiều)\s*(?:\S+\s+){0,5}\s*(?:biệt thự|villa)', # Bên cạnh là khu villa
                    r'(?:làm|xây|cải tạo)\s*(\S+\s+){0,4}(?:biệt thự|villa)', # Có thể xây thành biệt thự
                    r'(?:nhà|phố|mặt tiền|tòa nhà|building|chuyên|kinh doanh|chdv|căn hộ dịch vụ|kdt|kđt|khu đô thị|(?:\+84|0)\s*(?:\d\s*){3,6}(?:\d\s*){0,3}|văn phòng|cao ốc|nhà cao tầng)(?:\s+\S+){0,5} (?:biệt thự|villa)', # Tránh giới thiệu về cò
                    r'(?:biệt thự|villa)(?:\s+\S+){0,5} (?:nhà|phố|mặt tiền|tòa nhà|building|chuyên|kinh doanh|chdv|căn hộ dịch vụ|kdt|kđt|khu đô thị|(?:\+84|0)\s*(?:\d\s*){3,6}(?:\d\s*){0,3}|văn phòng|cao ốc|nhà cao tầng)', # Tránh giới thiệu về cò
                    r'(?:mua|xây) (\S+\s+){0,3}(?:biệt thự|villa)', # Loại các trường bán để chuyển qua mua hoặc xây biệt thự
                    r'(?:như|khu|toàn)\s*(?:\S+\s+){0,1}(?:biệt thự|villa)', # Các trường hợp đẹp như biệt thự, khu biệt thự
                    r'(?:ra|chuyển)\s*(?:\S+\s+){0,2}(?:biệt thự|villa)' # Chuyển ra để ở khu villa
                ]
                for pattern in villa_pattern:
                    if re.search(pattern, description):
                        if check_basement(description):
                            return 12_848_184 # Biệt thự có hầm
                        return 10_510_920 # Biệt thự không hầm
                for pattern in not_villa_pattern:
                    if re.search(pattern, description):
                        if row['Số tầng công trình'] == 2:
                            if check_basement(description):
                                return 6_275_876 # Nhà 1 tầng 1 hầm
                            else:
                                return 8_221_171 # Nhà 2 tầng không hầm
                        if row['Số tầng công trình'] < 2:
                            if check_basement(description):
                                return 6_275_876 # ví dụ như nhà 1.5 thì 0.5 đó chính là tầng hầm 
                            return 8_221_171 # Nhà 2 tầng không hầm
                        if check_basement(description):
                            return 9504604 # Nhà hơn 2 tầng, có hầm
                        return 8_221_171 # Nhà hơn 2 tầng, không hầm
                if check_basement(description):
                    return 12_848_184 # Biệt thự có hầm
                return 10_510_920 # Biệt thự không hầm
            if row['Số tầng công trình'] == 2:
                if check_basement(description):
                    return 6_275_876 # Nhà 1 tầng 1 hầm
                else:
                    return 8_221_171 # Nhà 2 tầng không hầm
            if row['Số tầng công trình'] < 2:
                if check_basement(description):
                    return 6_275_876 # ví dụ như nhà 1.5 thì 0.5 đó chính là tầng hầm 
                return 8_221_171 # Nhà 2 tầng không hầm
            if check_basement(description):
                return 9_504_604 # Nhà hơn 2 tầng, có hầm
            return 8_221_171 # Nhà hơn 2 tầng, không hầm

    @staticmethod
    def extract_width(row):
        if pd.notna(row.get("other_info", "{}")):
                try:
                    width = json.loads(row["other_info"]).get("Mặt tiền")
                    if pd.notna(width):
                        return DataCleaner._parse_and_clean_number(width)
                except (json.JSONDecodeError, TypeError):
                    pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower().strip() 
        text = DataCleaner.clean_description_text(text)
        
        if isinstance(text, str) and pd.notnull(text):

            match = re.search(r"(?<!cách\s)(?<!gần\s)(?<!rất\sgần\s)(mt|mặt tiền)\s*(\d+[.,]?\d*)\s*(mét|m)\b", text, re.IGNORECASE)
            if match:
                return float(match.group(2).replace(',', '.'))

            m = re.search(r"(?:diện\s*tích|dt)\s*:?\s*(\d+[.,]?\d*)\s*[x×]\s*(\d+[.,]?\d*)\s*(m|mét)?\b", text, re.IGNORECASE)
            if m:
                num1 = float(m.group(1).replace(',', '.'))
                num2 = float(m.group(2).replace(',', '.'))
                return min(num1, num2)
            
        return None 
    
    @staticmethod
    def extract_length(row):
        text = f"{row.get('title', '')} {row.get('description', '')}".lower().strip() 
        text = DataCleaner.clean_description_text(text)

        if isinstance(text, str) and not pd.isnull(text):
            m = re.search(r"(?:diện\s*tích|dt)\s*:?\s*(\d+[.,]?\d*)\s*[x×]\s*(\d+[.,]?\d*)\s*(m|mét)?\b", text, re.IGNORECASE)
            if m:
                num1 = float(m.group(1).replace(',', '.'))
                num2 = float(m.group(2).replace(',', '.'))
                width = DataCleaner.extract_width(row)

                if width is not None:
                    if abs(width - num1) < 0.1:
                        return num2
                    elif abs(width - num2) < 0.1:
                        return num1
                    else:
                        return max(num1, num2)
                else:
                    return max(num1, num2)

        return None
    
    @staticmethod 
    def extract_land_use(row):
        text = DataCleaner.clean_description_text(f"{row.get('title', '')} {row.get('description', '')}".lower().strip() )

        if "đất trồng" in text and "nhà" in text:
            return "Đất hỗn hợp"
        return "Đất ở"
        
    @staticmethod
    def extract_construction_area(row):
        text = DataCleaner.clean_description_text(str(row.get("description") or "").lower().strip())
        land_area = DataCleaner.extract_total_area(row)

        # Tìm diện tích thổ cư/đất ở trong bài theo từ khóa              
        m = r"\b(\d{1,3}(?:[.,]\d{1,2})?)\s?m(?:²|2)(?:\s?\([^)]+\))?\s?(?:full\s)?(?:thổ cư|đất ở)"
        m2 = r"(?:thổ cư|đất ở|full\s?thổ cư)[\s:.,\-–]{0,3}(\d{1,3}(?:[.,]\d{1,2})?)\s?m(?:²|2)"

        if m in text:
            return round(float(m.group(1).replace(",", ".")), 2)
        elif m2 in text:
            return round(float(m2.group(1).replace(",", ".")), 2)

        # Trong trường hợp không tìm được diện tích thổ cư/đất ở cụ thể trong bài, mặc định diện tích xây dựng = diện tích đất
        kws = ["full thổ cư", "thổ cư 100%", "thổ cư hoàn toàn", "toàn bộ thổ cư", "thổ cư toàn bộ"]
        for kw in kws:
            if kw in text:
                return land_area

        matches = re.findall(r"\b\w+\s+thổ cư\s+\w+\b", text, re.IGNORECASE)
        for match in matches:
            for kw in kws:
                score = fuzz.ratio(match.lower(), kw.lower())
                if score > 55:
                    return land_area       
        
        not_residential_land = ["đất ở, đất vườn"]
        for keyword in not_residential_land:
            if keyword not in text:
                return land_area
            
        return None 

    @staticmethod
    def extract_building_area(row):
        construction_area = DataCleaner.extract_construction_area(row)
        num_floors = DataCleaner.extract_num_floors(row)
        text = DataCleaner.clean_description_text(str(row.get("description") or "").lower().strip())
        match = re.search(r"(?i)diện\s+tích\s+sàn\s*:?\s*(\d+(?:[.,]\d+)?)\s*m[²2]", text, re.IGNORECASE)

        if match:
            return float(match.group(1).replace(",", "."))
        else:
            if construction_area is not None and num_floors is not None:
                return float(round(construction_area * num_floors, 2)) 
            return None 

    @staticmethod 
    def extract_adjacent_lane_width(row): 
        other_info = row.get("other_info", {})
        if isinstance(other_info, str):
            other_info = json.loads(other_info)

        val = other_info.get("Đường vào", None)
        text = DataCleaner.clean_description_text(f"{row.get('title', '')} {row.get('description', '')}".lower().strip())

        if val:
            try:
                num_val = float(str(val).replace(",", ".").strip().split()[0])
                if num_val <= 20:
                    return num_val
            except ValueError:
                pass  

        exact_match = re.search(
            r"\b(?:ngõ|đường|hẻm|ngách|kiệt)(?:\s+vào|\s+trước\s+nhà)?(?:\s+rộng|:)?\s*(\d+(?:[.,]\d+)?)(?=\s*(?:mét|m)\b)",
            text,
            re.IGNORECASE
        )
        if exact_match:
            return float(exact_match.group(1).replace(',', '.'))
        
        another_match = re.search(
            "\b(?:trước|vào)\s+nhà\s+rộng\s+(\d+(?:[.,]\d+)?)\s*(?:m|mét)\b",
            text,
            re.IGNORECASE
        )
        if another_match:
            return float(another_match.group(1).replace(",", "."))
            
        soft_match = re.search(
            r"\b(đường|ngõ|hẻm|ngách|kiệt)\b(?:\s+\S+){0,5}?\s*(\d+(?:[.,]\d+)?)(?=\s*(m|mét)\b)", # Bắt những cụm từ như "đường rộng 10m", "hẻm 4m",...
            text,
            re.IGNORECASE
        )
        if soft_match:
            start_index = soft_match.start()
            before_text = text[:start_index]
            last_3_words = re.findall(r"\b\w+\b", before_text)[-3:]
            blacklist = {"cách", "ra", "tới"} # Loại bỏ những cụm như "cách đường Phạm Hùng 200m", "tới hẻm thông 10m",...
            num = float(soft_match.group(2).replace(",", "."))

            if not any(word.lower() in blacklist for word in last_3_words) and num <= 35:
                return num 
            
        best_match = process.extractOne(
        query=text,
        choices=ALLEY_WIDTH.keys(),
        scorer=fuzz.partial_ratio
    )
    
        if best_match:
            matched_text, score, _ = best_match
            if score >= 90: 
                return ALLEY_WIDTH[matched_text]
                
        return None             

    @staticmethod
    def extract_distance_to_the_main_road(row):
        val = None
        try:
            other_info = row.get("other_info")
            if other_info:
                other_info_dict = json.loads(other_info)
                val = other_info_dict.get("Đường vào")
        except (json.JSONDecodeError, TypeError, AttributeError):
            val = None

        # Try to parse val to number (meters)
        num_val = None
        if val is not None:
            try:
                # Remove spaces and units, replace comma with dot
                val_str = str(val).lower().replace(" ", "").replace(",", ".")
                if val_str.endswith("km"):
                    num_val = float(val_str.replace("km", "")) * 1000
                elif val_str.endswith("m"):
                    num_val = float(val_str.replace("m", ""))
                else:
                    num_val = float(val_str)
            except ValueError:
                num_val = None

        text = DataCleaner.clean_description_text(f"{row.get('title', '')} {row.get('description', '')}".lower().strip())
        lat, lon = row.get("latitude", ""), row.get("longitude", "")

        if num_val is not None and num_val >= 50:
            return num_val
        elif pd.notna(text):
            # TH1: Nhà nằm trên mặt phố 
            if DataCleaner._is_on_main_road(text, lat, lon) == "Mặt phố":
                return 0 
            
            # TH2: Ước lượng gần phố (nhà, bước chân, phút,...)
            major_roads = 'đường|phố|mặt đường|mặt phố|mp|vành đai|đại lộ|đl'
            tertiary = 'mặt tiền|mt|trục chính|quốc lộ|ql|qlo|tỉnh lộ|tl|cầu|ngã tư|ngã ba|ngã 4|ngã 3|nhà'
            landmarks = 'trường|chợ|siêu thị|vincom|aeon|lotte|công viên|cv|hẻm|hxh|ngõ|vườn|trung tâm|vinmart|winmart|vin|mall|tttm|bigc|go|gigamall|đại học|đh|bến xe|bx|ga'
            places_of_interest = 'biển|sông|hồ|ubnd|chung cư|cc|khu đô thị|kđt|kdt|sân bay|bệnh viện|bv|quận|q(?:\d+)|thành phố|tp|huyện|thị xã|thị trấn|tx|bán kính'

            # Cách bao nhiêu căn nhà ra mặt phố 
            if re.search(rf'cách\s*(?:\S+\s*){{0,2}}nhà\s*(?:\S+\s*){{0,3}}(?:{major_roads}|{tertiary})', text): 
                return 30 
            
            # Cách bao nhiêu bước chân ra phố, tới phố có bao nhiêu bước chân...
            pattern = re.search(rf'(bước\s*(?:\S+\s*){{0,5}}ra\s*(?:{major_roads}|{tertiary}))|(cách\s*(?:\S+\s*){{0,2}}(?:{major_roads}|{tertiary})\s*(?:\S+\s*){{0,2}}bước)', text)
            if pattern:
                return 5
            
            # Cách bao nhiêu phút ra phố 
            if re.search(rf'phút(?:\S+\s*){{0,3}}ra\s*(?:{major_roads}|{tertiary}\s*)', text):
                return 50
            
            # Gần, sát, giáp phố 
            if re.search(rf'(?:gần|giáp|sát)(?:\s+\S+){{0,2}}\s+(?:{major_roads}|{tertiary})', text):
                return 20
            
            # TH3: Ngõ nông
            if re.search(r'ngõ\s*(?:\S+\s*){0,3}(?:nông|ngắn)', text):
                return 10
            
            # TH4: Cách đường bao nhiêu m/bao nhiêu mét ra mặt đường
            match_1 = re.search(rf'cách\s*(?:{major_roads})\s*(?:(?!\d{{1,3}}(?:[.,]\d+)*\s*(?:m|km))\S+\s*){{0,5}}?\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # cách đường Phạm Văn Đồng 5m, cách mặt phố Hai Bà Trưng 100m 
            match_2 = re.search(rf'\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))\s*ra\s*(?:{major_roads})\s*(?:\S+\s+){{0,5}}', text, re.IGNORECASE) # 50m ra đường Cầu Giấy
            match_3 = re.search(rf'ra\s*(?:{major_roads}|{tertiary})\s*(?:\S+\s+){{0,5}}\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # ra mặt phố cổ 20 m
            match_4 = re.search(rf'cách\s*(?:{major_roads})\s*(?:(?!\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))(?!{landmarks}|{places_of_interest})\S+\s*){{1,7}}\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # cách mặt tiền Trần Nhân Tông 20m
            match_5 = re.search(rf'(?:{major_roads}|{tertiary})(?!\s+(?:{landmarks}|{places_of_interest}))\s*(?:\b\w+\b\W+){{0,5}}?cách\s*(?:\b\w+\b\W+){{0,5}}(\d+(?:[.,]\d+)*\s*(?:m|km))', text, re.IGNORECASE) # đường Trường Chinh cách nhà 25m 
            match_6 = re.search(rf'\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))\s*ra\s*(?:\S+\s*){{0, 2}}(?:{tertiary})\s*(?:\S+\s+){{0,5}}', text, re.IGNORECASE) # 50m ra Quốc Lộ 1A
            match_7 = re.search(rf'\D(\d{{1,3}}(?:[,.]\d+)?\s*(?:m|km))\s*ra\s*(?:(?!{landmarks}|{places_of_interest})\S+\s+)', text, re.IGNORECASE) # 28m ra Trần Hưng Đạo 
            
            all_patterns = [match_1, match_2, match_3, match_4, match_5, match_6, match_7]
           
            for pattern in all_patterns:
                if pattern:
                    if re.search(landmarks, pattern.group(0)) or re.search(places_of_interest, pattern.group(0)):
                        continue 
                    if 'km' in pattern.group(1):
                        result = float(pattern.group(1).replace('km', '').replace(',', '.'))
                        result *= 1000
                        return result
                    return float(pattern.group(1).replace('m', '').replace(',', '.'))
                    
            greedy_matches = re.search(r'(?:\b\w+\b\W+){0,5}?cách\s*(?:(?!\d+[.,])\S+\s*){0,7}\D(\d{1,3}(?:[.,]\d+)?\s*k*m)(?:\S+\s*){0,7}', text, re.IGNORECASE)
            if greedy_matches:
                if re.search(landmarks, greedy_matches.group(0)) or re.search(places_of_interest, greedy_matches.group(0)):
                    pass
                else:
                    if 'km' in greedy_matches.group(1):
                        result = float(greedy_matches.group(1).replace('km', '').replace(',', '.'))
                        result *= 1000
                        return result
                    return float(greedy_matches.group(1).replace('m', '').replace(',', '.'))
            
        return None 


class DataImputer:
    _graph_cache = {}

    @staticmethod
    def fill_missing_width(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing width by looking up in a ground truth file.
        """
        target_col = 'Kích thước mặt tiền (m)'

        try:
            ground_truth = pd.read_excel(INFO_FILE)
            print(f"Loaded ground truth file successfully from: {INFO_FILE}")
        except FileNotFoundError:
            print(f"Ground truth file not found.")
            return df
        
        required_cols = ['Tỉnh/Thành phố', 'Diện tích (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài']
        if not all(col in ground_truth.columns for col in required_cols):
            print(f"Missing columns in ground truth file: {required_cols}.")
            return df

        ground_truth.dropna(subset=required_cols, inplace=True)
        ground_truth = ground_truth[(ground_truth[target_col] > 0) & (ground_truth['Kích thước mặt tiền (m)'] > 0)]

        if ground_truth.empty:
            print("Cảnh báo: Không có dữ liệu hợp lệ trong file ground truth sau khi làm sạch. Bỏ qua.")
            return df

        df_imputed = df.copy()
        rows_to_impute = df_imputed[df_imputed[target_col].isna()].index

        for index in rows_to_impute:
            row = df_imputed.loc[index]
            target_province = row['Tỉnh/Thành phố']
            target_area = row['Diện tích đất (m2)']

            if pd.isna(target_province) or pd.isna(target_area):
                continue

            # 1. Filter ground truth file based on locations 
            gt_province_subset = ground_truth[ground_truth['Tỉnh/Thành phố'] == target_province].copy()
            
            if gt_province_subset.empty:
                continue

            # 2. Look up the estate with closest area
            gt_province_subset.loc[:, 'area_diff'] = (gt_province_subset['Diện tích (m2)'] - target_area).abs()
            best_match = gt_province_subset.loc[gt_province_subset['area_diff'].idxmin()]
            gt_width = best_match['Kích thước mặt tiền (m)']
            gt_length = best_match['Kích thước chiều dài']

            # 3. Calculate the width from the ground truth estate's shape ratio 
            shape_ratio = gt_width / gt_length
            imputed_width = (target_area * shape_ratio) ** 0.5
            df_imputed.loc[index, target_col] = round(imputed_width, 2)
            imputed_count += 1

        return df_imputed

    @staticmethod
    def fill_missing_length(row):
        """
        Fills missing length by dividing the construction area by the width.
        Uses already calculated columns.
        """
        length = row.get("Kích thước chiều dài (m)")

        if pd.isna(length):
            width = row.get("Kích thước mặt tiền (m)")
            construction_area = row.get('Diện tích xây dựng')
            
            # Ensure both width and area are valid numbers before dividing
            if pd.notna(width) and width > 0 and pd.notna(construction_area):
                return round(float(construction_area) / float(width), 2)
        
        # If length is not missing, or if it cannot be imputed, return the original value.
        return length 

    @staticmethod
    def _get_cached_graph(lat, lon, radius=500):
        """
        Retrieve a cached OSMnx graph if nearby, otherwise download a new one.
        """
        key = (round(lat, 4), round(lon, 4), radius)
        if key in DataImputer._graph_cache:
            return DataImputer._graph_cache[key]

        try:
            G = ox.graph_from_point((lat, lon), dist=radius, network_type="drive")
            DataImputer._graph_cache[key] = G
            return G
        except Exception as e:
            print(f"OSMnx graph fetch failed for ({lat}, {lon}): {e}")
            return None

    @staticmethod
    def _distance_to_named_street(lat, lon, street_name, radius_m=500):
        """
        Compute distance (m) from a point to a street with the given name,
        ignoring alleys or side lanes (hẻm, ngõ, ngách...).
        """
        G = DataImputer._get_cached_graph(lat, lon, radius_m)
        if G is None:
            return None

        try:
            gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            roads = gdf_edges[gdf_edges["name"].notna()].copy()
            if roads.empty:
                return None

            # Filter by name (case-insensitive)
            roads = roads[roads["name"].str.lower().str.contains(street_name.lower(), na=False)]
            if roads.empty:
                return None

            # Exclude alleys and small lanes
            pattern = r"\b(?:hẻm|hem|ngách|ngõ|ngo|alley|ngõ\s*\d+|hẻm\s*\d+|ngách\s*\d+)\b"
            roads = roads[~roads["name"].str.lower().str.contains(pattern, flags=re.IGNORECASE, na=False)]
            if roads.empty:
                return None

            # Reproject for accurate meter-based distance
            roads_proj = roads.to_crs(epsg=3857)
            point_proj = ox.projection.project_geometry(Point(lon, lat), to_crs=roads_proj.crs)[0]

            roads_proj["dist"] = roads_proj.distance(point_proj)
            nearest = roads_proj.loc[roads_proj["dist"].idxmin()]
            min_dist_m = nearest["dist"]

            # If already on or very close to the road
            if min_dist_m <= 5:
                return {"street_name": nearest.get("name"), "distance_m": 0.0}

            return {"street_name": nearest.get("name"), "distance_m": round(min_dist_m, 1)}

        except Exception as e:
            print(f"Distance computation failed for {street_name}: {e}")
            return None

    @staticmethod
    def _distance_to_main_road(lat, lon, radius_m=800):
        """
        Compute distance to the nearest major road (by highway tag).
        """
        G = DataImputer._get_cached_graph(lat, lon, radius_m)
        if G is None:
            return None

        try:
            gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            if gdf_edges.empty:
                return None

            main_tags = ["motorway", "trunk", "primary", "secondary", "tertiary"]
            main_roads = gdf_edges[gdf_edges["highway"].apply(
                lambda h: any(tag in (h if isinstance(h, list) else [h]) for tag in main_tags)
            )]

            if main_roads.empty:
                return None

            main_roads = main_roads.to_crs(epsg=3857)
            point_proj = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

            min_dist_m = main_roads.distance(point_proj).min()
            return round(min_dist_m, 1)

        except Exception as e:
            print(f"Main-road distance computation failed: {e}")
            return None

    @staticmethod
    def fill_missing_distance_to_the_main_road(row):
        """
        Fill 'Khoảng cách tới trục đường chính (m)'.
        """

        def normalize_street_name(name: str) -> str:
            """Normalize Vietnamese street names for OSM matching."""
            if not isinstance(name, str) or not name.strip():
                return ""
            prefixes = [
                r"\bđường\b", r"\bphố\b", r"\bđại\s*lộ\b", r"\btỉnh\s*lộ\b",
                r"\bquốc\s*lộ\b", r"\bql\d+\b", r"\bhẻm\b", r"\bngách\b", r"\bngõ\b"
            ]
            cleaned = re.sub(r"^(?:" + "|".join(prefixes) + r")\s*", "", name, flags=re.IGNORECASE).strip()
            return re.sub(r"\s+", " ", cleaned).lower()

        # Already has distance
        if pd.notna(row.get("Khoảng cách tới trục đường chính (m)")):
            return row["Khoảng cách tới trục đường chính (m)"]

        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            return None

        street_name = row.get("Đường phố")
        if isinstance(street_name, str) and street_name.strip():
            street_name = normalize_street_name(street_name)
            result = DataImputer._distance_to_named_street(lat, lon, street_name)
            if result:
                print(f"{street_name}: {result['distance_m']} m (named street)")
                return result["distance_m"]

        # Fallback to OSM main-road tags
        dist_main = DataImputer._distance_to_main_road(lat, lon)
        if dist_main is not None:
            print(f"Main-road fallback: {dist_main} m")
            return dist_main

        # Random fallback
        fallback = random.randint(20, 200)
        print(f"No OSM or named match → fallback {fallback}m")
        return fallback
    
    # @staticmethod
    # def fill_missing_adjacent_lane_width_for_large_dataset(df):
    #     """
    #     Fill missing adjacent lane width by using proxy price classified by location. 
    #     Large dataset remix. 
    #     """
    #     target_col = 'Độ rộng ngõ/ngách nhỏ nhất (m)'
    #     proxy_col = 'Đơn giá đất'
    #     location_hierarchy = [
    #         'Đường phố', 
    #         'Xã/Phường/Thị trấn', 
    #         'Thành phố/Quận/Huyện/Thị xã', 
    #         'Tỉnh/Thành phố'
    #     ] # from most detailed to most general 

    #     df_onehousing = pd.read_excel(ONEHOUSING_FILE, header=True)
    #     df_imputed = df.copy()

    #     # Concat the data from OneHousing to have a bigger data lake to look up the price 
    #     df_filtered = df_onehousing[df_imputed.columns.intersection(df_imputed.columns)]
    #     result = pd.concat([df_imputed, df_filtered], ignore_index=True)
    #     df_known = result[result[target_col].notna() & result[proxy_col].notna()].copy()

    #     if len(df_known) < 20: # Minimum threshold to infer the adjacent lane width from price 
    #         print("Warning: Not enough data. Skipping...")
    #         return df
        
    #     # 1. Split the adjacent lane width into different bins 
    #     bins = [0, 2, 3.5, 5, float('inf')]
    #     labels = ['Ngách/Hẻm', 'Ngõ nhỏ', 'Ngõ lớn', 'Đường chính']
    #     df_known['width_category'] = pd.cut(df_known[target_col], bins=bins, labels=labels, right=False)

    #     # 2. Calculate the median price for each location level 
    #     medians_cache = {}
    #     for i in range(len(location_hierarchy) + 1):
    #         group_cols = location_hierarchy[i:] + ['width_category']
    #         group_cols = [col for col in group_cols if col != 'width_category' and col in df_known.columns] + ['width_category']
    #         level_name = "_".join(location_hierarchy[i:]) if i < len(location_hierarchy) else "global"
    #         medians_cache[level_name] = df_known.groupby(group_cols, observed=True)[proxy_col].median()

    #     representative_widths = {
    #         'Ngách/Hẻm': 1.5, 'Ngõ nhỏ': 3,
    #         'Ngõ lớn': 4.0, 'Đường chính': 7.0 
    #     }

    #     def _find_best_width(row):
    #         """
    #         Function to find the best adjacent lane width. 
    #         """
    #         price_to_compare = row[proxy_col]
    #         if pd.isna(price_to_compare):
    #             return None

    #         # Iterate through the location hierarchy 
    #         for i in range(len(location_hierarchy) + 1):
    #             level_name = "_".join(location_hierarchy[i:]) if i < len(location_hierarchy) else "global"
                
    #             # # Create a key to look up price in the cache 
    #             loc_key = tuple(row[col] for col in location_hierarchy[i:] if col in row.index)
    #             current_medians = medians_cache[level_name]

    #             try:
    #                 if level_name != "global":
    #                     local_price_medians = current_medians.loc[loc_key]
    #                 else:
    #                     local_price_medians = current_medians 

    #                 if not local_price_medians.empty and local_price_medians.notna().any():
    #                     distances = (local_price_medians - price_to_compare).abs()
    #                     closest_category = distances.idxmin()
    #                     return representative_widths[closest_category]
    #             except (KeyError, TypeError):
    #                 continue
            
    #         return None 
        
    #     # 4. Fill in missing values 
    #     impute_mask = df_imputed[target_col].isna() & df_imputed[proxy_col].notna()
    #     imputed_values = df_imputed[impute_mask].apply(_find_best_width, axis=1)
    #     df_imputed.loc[impute_mask, target_col] = imputed_values 
    #     print(f"Successfully filled in {imputed_values.notna().sum()} missing adjacent lane width values.")
    #     return df_imputed


class FeatureEngineer:
    @staticmethod
    def calculate_estimated_price(row):
        """
        Calculate the estimated price of the property.
        """
        total_price = row.get('Giá rao bán/giao dịch')

        if not total_price:
            return None

        return round(total_price * 0.98, 2)
    
    @staticmethod
    def get_location_category(row):
        """
        Determines the asset's position category (VT1-VT4) based on distance
        to the main road and alley width.
        """
        distance = row.get('Khoảng cách tới trục đường chính (m)')
        alley_width = row.get('Độ rộng ngõ/ngách nhỏ nhất (m)')

        # Cannot determine VT without distance to the main road.
        if pd.isna(distance):
            return None

        if distance == 0:
            return "VT1"

        # If distance > 0, alley width is required to classify further.
        if pd.isna(alley_width):
            return None

        if alley_width >= 3.5:
            return "VT2"
        elif 2 <= alley_width < 3.5:
            return "VT3"
        elif alley_width < 2:
            return "VT4"

        return None

    def calculate_business_advantage(row):
        """
        Calculates the 'Lợi thế kinh doanh' (Business Advantage) based on
        the asset's location category (VT) and its district type (Quận/Huyện).
        """
        location_cate = FeatureEngineer.get_location_category(row)
        district_name = row.get('Thành phố/Quận/Huyện/Thị xã')

        if not location_cate or not isinstance(district_name, str):
            return "Kém"

        is_quan = "quận" in district_name.lower()
        is_huyen = not is_quan

        if location_cate in ["VT1", "VT2"] and is_quan:
            return "Tốt"
        if location_cate == "VT1" and is_huyen:
            return "Khá"
        if (location_cate == "VT2" and is_huyen) or (location_cate == "VT3"):
            return "Trung bình"

        return "Kém"

    @staticmethod
    def calculate_land_unit_price(row):
        """
        Calculates the land unit price per square meter.
        - If 'is_land' is True, uses a simple formula: Price / Area.
        - Otherwise, subtract building value from the price before dividing with area.
        """
        # --- Use the temporary boolean flag ---  
        total_price = row.get('Giá rao bán/giao dịch')
        land_area = row.get('Diện tích đất (m2)')
        construction_cost_per_sqm = row.get('Đơn giá xây dựng')
        remaining_quality = row.get('Chất lượng còn lại')
        total_floor_area = row.get('Tổng diện tích sàn')

        if pd.isna(total_price) or pd.isna(construction_cost_per_sqm) or \
                pd.isna(land_area) or pd.isna(remaining_quality) or \
                pd.isna(total_floor_area):
            return None

        if land_area <= 0 or total_floor_area < 0:
            return None

        building_value = construction_cost_per_sqm * total_floor_area * remaining_quality

        if building_value >= total_price:
            return round(total_price / land_area, 2)

        land_value = total_price - building_value
        land_unit_price = (land_value * 0.98) / land_area

        return round(land_unit_price, 2)