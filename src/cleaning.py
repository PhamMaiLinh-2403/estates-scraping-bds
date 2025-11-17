import re
import json
import time 
import pandas as pd
import numpy as np
import random
import osmnx as ox
import networkx as nx
from rapidfuzz import fuzz, process
from rapidfuzz.fuzz import ratio
from math import * 
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
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
        negation_keywords = ['không', 'chưa', 'ko', 'chẳng', 'khó']
        window = 4  

        for neg in negation_keywords:
            pattern = rf'\b{neg}(?:\s+\w+){{0,{window}}}?\s+{re.escape(keyword)}\b'
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
            r'(lh|liên hệ|zalo|sđt|phone|call|chuyên bđs|số điện thoại|ký gửi|kí gửi)',
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
    def _is_on_main_road(text: str, short_address, lat, lon):
        text = DataCleaner.clean_description_text(text.lower().strip())

        # geolocator = Nominatim(user_agent="vn_address_distance")

        not_on_main_road = r"mặt ngõ|mặt hẻm|gần phố|sát phố|nhà hẻm|nhà ngõ|ngõ vào|ngõ thông|hẻm vào|hẻm thông|hxh|hxt|(?<=\d)sẹc|(?<=\d)xẹt|hẻm|ngõ|ngách|kiệt ô tô|kiệt xe máy|kiệt(?<=\d)"
        major_roads = 'đường|phố|mặt đường|mặt phố|mp|vành đai|đại lộ|đl|mặt tiền|mt|trục chính|quốc lộ|ql|qlo|tỉnh lộ|tl|ngã tư|ngã ba|ngã 4|ngã 3'
        on_main_road = "\b(?:ngay|trên|nằm)?\s*(?:mặt tiền|mặt phố|mặt đường|phố|đường)\b"
        distance_to_main_road = r"(?:vài\s*bước|[0-9]+\s*m|[0-9]+m|mấy\s*m|chỉ\s*\d+\s*m)(?:\s+\w+){0,3}\s+(?:ra|tới)\s+(?:mặt\s*tiền|đường|phố|đường\s+chính|mặt\s*phố)"

        if re.search(not_on_main_road, text):
            return "Mặt ngõ"
        elif re.search(rf'(?:cách|ra|gần|sát)(?:\s+\w+){{0,5}}\s+(?:{major_roads})', text, re.IGNORECASE):
            return "Mặt ngõ"
        elif re.search(distance_to_main_road, text, re.IGNORECASE):
            return "Mặt ngõ"
        elif re.search(on_main_road, text, re.IGNORECASE):
            return "Mặt phố"
        # elif lat is not None and lon is not None:
        #     lat, lon = float(lat), float(lon)
        #     try:
        #         location = geolocator.reverse((lat, lon), language="vi", timeout=10)
        #         print(f"Successfully completed reverse geocoding for coordinates: ({lat}, {lon})")
        #         time.sleep(random.uniform(1, 2))  

        #         if location and location.address:
        #             address = location.address.lower()
        #             for loc in ["ngõ", "hẻm", "ngách"]:
        #                 if loc in address:
        #                     return "Mặt ngõ"
                
        #         if re.search(on_main_road, text) or "nhà phố" in text.lower():
        #             return "Mặt phố"

        #     except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
        #         print(f"Reverse geocoding request failed for ({lat}, {lon}): {e}")
        #         time.sleep(1) 

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
                match = re.search(r'\b(đường|phố|quốc lộ|đại lộ|xa lộ|tỉnh lộ)\s+[\w\s\-()\/]+', part.lower())
                if match and len(match.group(0).split()) <= 5:
                    return match.group(0).title().strip()

        parts_raw = str(row.get("address_parts", "")).strip()
        if parts_raw:
            try:
                addr_list = json.loads(parts_raw)
                if addr_list:
                    last_item = addr_list[-1]
                    m = re.search(r"tại (đường|phố|quốc lộ|đại lộ|xa lộ|tỉnh lộ)\s+([^,]+)", last_item, re.IGNORECASE)
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
        ignore_prefixes = ['đường', 'phố', 'phường', 'quận', 'huyện', 'quốc lộ', 'đại lộ']

        for part in parts:
            lower_part = part.lower().strip()

            # Skip if it starts with an ignore prefix followed by "số"
            if any(lower_part.startswith(f"{prefix} số") for prefix in ignore_prefixes):
                continue

            if any(re.match(rf"^{prefix}\s+\d+([a-z]*)\b", lower_part) for prefix in ignore_prefixes):
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
        
        return DataCleaner._is_on_main_road(text, short_address, lat, lon)
            
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
            if re.search(pattern = r'nhà cấp 4|nhà c4|cấp 4\W|nc4|nhà trệt|nhà nát|c4', string=text):
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

            match = re.search(r"(?<!cách\s)(?<!gần\s)(?<!rất\sgần\s)(?<!tới\s)(?<!đến\s)(mt|mặt tiền|rộng)\s*(\d+[.,]?\d*)\s*(mét|m)\b", text, re.IGNORECASE)
            if match:
                return float(match.group(2).replace(',', '.'))

            # m = re.search(r"(?:diện\s*tích|dt)\s*:?\s*(\d+[.,]?\d*)\s*[x×]\s*(\d+[.,]?\d*)\s*(m|mét)?\b", text, re.IGNORECASE)
            m = re.search(r"\(?\s*(\d+[.,]?\d*)\s*[x×]\s*(\d+[.,]?\d*)\s*\)?\s*(m|mét)?\b", text, re.IGNORECASE)
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
        
        not_residential_land = ["đất trồng, đất vườn"]
        for keyword in not_residential_land:
            if keyword not in text:
                return land_area
            
        return None 

    @staticmethod
    def extract_building_area(row):
        construction_area = DataCleaner.extract_construction_area(row)
        num_floors = DataCleaner.extract_num_floors(row)
        text = DataCleaner.clean_description_text(str(row.get("description") or "").lower().strip())

        patterns = [
            r"(?i)diện\s+tích\s+sàn(?:\s+\S+){0,5}?\s+(\d+(?:[.,]\d+)?)\s*m[²2]",
            r"(?i)\bsàn(?:\s+\S+){0,5}?\s+(\d+(?:[.,]\d+)?)\s*m[²2]",
            r"(?i)(\d+(?:[.,]\d+)?)\s*m[²2](?:\s+\S+){0,5}?\s+diện\s+tích\s+sàn",
            r"(?i)(\d+(?:[.,]\d+)?)\s*m[²2](?:\s+\S+){0,5}?\s+sàn",
            r"(?i)diện\s+tích\s+sử\s+dụng(?:\s+\S+){0,5}?\s+(\d+(?:[.,]\d+)?)\s*m[²2]",
            r"(?i)(\d+(?:[.,]\d+)?)\s*m[²2](?:\s+\S+){0,5}?\s+diện\s+tích\s+sử\s+dụng",
            r"(?i)dtsd(?:\s+\S+){0,5}?\s+(\d+(?:[.,]\d+)?)\s*m[²2]",
            r"(?i)(\d+(?:[.,]\d+)?)\s*m[²2](?:\s+\S+){0,5}?\s+dtsd"
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1).replace(",", "."))
                except ValueError:
                    continue  

        if construction_area is not None and num_floors is not None:
            return round(construction_area * num_floors, 2)

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
                if num_val <= 15:
                    return num_val
            except ValueError:
                pass  
            
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
        short_address = row.get("short_address", "").lower().strip()
        lat, lon = row.get("latitude", ""), row.get("longitude", "")

        if num_val is not None and num_val >= 50:
            return num_val
        elif pd.notna(text):
            # TH1: Nhà nằm trên mặt phố 
            if DataCleaner._is_on_main_road(text, short_address, lat, lon) == "Mặt phố":
                return 0 
            
            # TH2: Cách đường bao nhiêu m/bao nhiêu mét ra mặt đường
            major_roads = 'đường|phố|mặt đường|mặt phố|mp|vành đai|đại lộ|đl'
            tertiary = 'mặt tiền|mt|trục chính|oto|ô tô'
            landmarks = 'trường|chợ|siêu thị|vincom|aeon|lotte|công viên|cv|hẻm|hxh|ngõ|vườn|trung tâm|vinmart|winmart|vin|mall|tttm|bigc|go|gigamall|đại học|đh|bến xe|bx|ga'
            places_of_interest = 'biển|sông|hồ|ubnd|chung cư|cc|khu đô thị|kđt|kdt|sân bay|bệnh viện|bv|quận|q(?:\d+)|thành phố|tp|huyện|thị xã|thị trấn|tx|bán kính'
            
            match_1 = re.search(rf'cách\s*(?:{major_roads})\s*(?:(?!\d{{1,3}}(?:[.,]\d+)*\s*(?:m|km))\S+\s*){{0,5}}?\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # cách đường Phạm Văn Đồng 5m, cách mặt phố Hai Bà Trưng 100m 
            match_2 = re.search(rf'\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))\s*ra\s*(?:{major_roads})\s*(?:\S+\s+){{0,5}}', text, re.IGNORECASE) # 50m ra đường Cầu Giấy
            match_3 = re.search(rf'ra\s*(?:{major_roads}|{tertiary})\s*(?:\S+\s+){{0,5}}\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # ra mặt phố cổ 20 m
            match_4 = re.search(rf'cách\s+(?:\S+\s+)*?(?:{major_roads})\s+(?:\S+\s*){{0,4}}?(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # cách mặt tiền Trần Nhân Tông 20m
            match_5 = re.search(rf'(?:(?:{major_roads}|{tertiary})\s+\S+\s*){{1,5}}cách\s+(?:\S+\s*){{0,5}}(\d+(?:[.,]\d+)?\s*(?:m|km))', text, re.IGNORECASE) # đường Trường Chinh cách nhà 25m 
            match_6 = re.search(rf'\D(\d{{1,3}}(?:[.,]\d+)?\s*(?:m|km))\s*ra\s*(?:\S+\s*){{0,2}}(?:{tertiary})\s*(?:\S+\s+){{0,5}}', text, re.IGNORECASE) # 50m ra Quốc Lộ 1A
            pattern_7 = re.findall(r'(\d{1,3}(?:[.,]\d+)?\s*(?:m|km))\s*ra\s*([\w\s]+)', text, re.IGNORECASE)

            match_7 = None
            for distance, place in pattern_7:
                if not re.search(rf'\b({landmarks}|{places_of_interest})\b', place, re.IGNORECASE):
                    class DummyMatch:
                        def __init__(self, distance, place):
                            self._distance = distance
                            self._place = place
                        def group(self, idx):
                            if idx == 0:
                                return f"{self._distance} ra {self._place}"
                            elif idx == 1:
                                return self._distance
                    match_7 = DummyMatch(distance, place)
                    break  

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
            
            # TH3: Ước lượng gần phố (nhà, bước chân, phút,...)
            # Cách bao nhiêu căn nhà ra mặt phố 
            if re.search(rf'cách\s*(?:\S+\s*){{0,2}}nhà\s*(?:\S+\s*){{0,3}}(?:{major_roads}|{tertiary})', text): 
                return 30 
            
            # Cách bao nhiêu bước chân ra phố, tới phố có bao nhiêu bước chân...
            pattern = re.search(rf'(bước\s*(?:\S+\s*){{0,5}}ra\s*(?:{major_roads}|{tertiary}))|(cách\s*(?:\S+\s*){{0,2}}(?:{major_roads}|{tertiary})\s*(?:\S+\s*){{0,2}}bước)', text)
            if pattern:
                return 10
            
            # Cách bao nhiêu phút ra phố 
            if re.search(rf'phút(?:\S+\s*){{0,3}}ra\s*(?:{major_roads}|{tertiary}\s*)', text):
                return 50
            
            # Gần, sát, giáp phố 
            if re.search(rf'(?:gần|giáp|sát)(?:\s+\S+){{0,2}}\s+(?:{major_roads}|{tertiary})', text):
                return 10
            
            if "ngõ nông" in text.lower():
                return 20 
            
        return None 
    
    @staticmethod
    def search_pho(string, short_add):
        string_lower = string.lower()
        main_road = '(?:\S+\s){0,5}(?:nhà(?:\s\S+){0,2} mặt phố|mặt phố|mặt tiền phố|mt phố|phân lô phố|nhà(?:\s\S+){0,2} mặt đường|nhà(?:\s\S+){0,2} đường|mặt đường| mt đường|mặt tiền đường|\Wmtd\W|\Wmtđ\W)\s?(?:\S+\s){0,4}'
        short_main_road = '(?:nhà(?:\s\S+){0,2} mặt phố|mặt phố|mặt tiền phố|mt phố|phân lô phố|nhà(?:\s\S+){0,2} mặt đường|nhà(?:\s\S+){0,2} đường|mặt đường| mt đường|mặt tiền đường|\Wmtd\W|\Wmtđ\W)'
        if len(re.findall(short_main_road, string_lower)) >= 5:
            return 'Drop'
        close = 'gần|cạnh|\Wra\W|sát|giáp|cách|sau|tránh|đối diện|kết nối|tương lai|quy hoạch|ký gửi|ký gởi|kí gửi|kí gởi|chuyên|nhà ngõ|nhà ngách|nhà hẻm|nhà kiệt|biệt thự|liên hệ|\Wlh\W|bước|căn|(?:vài|\d+)\snhà|phút|\d+p'
        if short_add != '':
            short_add_split = short_add.lower().strip().split(',')
            if 'đường' in short_add_split[0] or 'phố' in short_add_split[0]:
                road_name_in_short_add = short_add_split[0].replace('đường', '').replace('phố', '').strip()
                if re.search(rf'(?:{close})\s?(?:\S+\s){{0,5}}{re.escape(road_name_in_short_add)}', string_lower):
                    return None
                if re.search(rf'{re.escape(road_name_in_short_add)}(?![^\.,\?!]*[.,\?!])\s?(?:\S+\s){{0,2}}(?:{close})', string_lower):
                    return None
        pho = re.search(main_road, string_lower)
        if pho:
            # Nếu các cụm đó chỉ là gần phố, gần đường abc --> thì bỏ
            if re.search(close, pho.group(0)):
                return None
            # return 'Mặt phố'
            else:
                # Tìm xem nó đang nằm trên mặt đường nào
                road = re.search(short_main_road, string_lower)
                # if road:
                road_span = road.span()
                # start = 0 if road_span[0] < 40 else road_span[0] - 40
                # Tìm 20 ký tự sau các keyword về mặt phố mặt ngõ
                end = len(string) if road_span[1] + 20 > len(string) else road_span[1] + 20
                road_name_list = string[road_span[1]:end].split()
                if len(road_name_list) < 1:
                    road_name = None
                elif len(road_name_list) == 1:
                    if road_name_list[0][0].isupper() or re.match(r'\d+/\d+', road_name_list[0]):
                        road_name = road_name_list[0]
                    else:
                        road_name = None
                else:
                    # Nếu hai từ đầu tiên sau đó được viết hoa thì khả năng cao nó chính là tên đường
                    if road_name_list[0][0].isupper() and road_name_list[1][0].isupper():
                        road_name = road_name_list[0]  + ' ' + road_name_list[1]
                    # Nếu từ đầu tiên có dạng kiểu 23/5 (đường 2/9)
                    elif re.match(r'\d+/\d+', road_name_list[0]):
                        road_name = road_name_list[0]
                    else:
                        road_name = None
                if road_name:
                    # Tìm sự xuất hiện của tên đường trong phần còn lại của description
                    search_string = f'(?:\S+\s){{0,5}}{re.escape(road_name)}\W?\s?(?:\S+\s){{0,4}}'
                    while True:
                        if len(string) <= 5:
                            if re.search('(?:gần|cạnh|cách|\Wra|giáp|sát) (?:phố|mặt phố)', string_lower) or re.search(r'(?:phố|mặt phố)\s(?:\S+\s)?(?:gần|cạnh|cách|ra\W|giáp|sát|vào)', string_lower):
                                return None
                            return 'Mặt phố'
                        road_appearance = re.search(search_string, string)
                        if road_appearance:
                            # Nếu có các từ trong trường gần ở xung quanh tên đường ở đằng sau
                            if re.search(close, road_appearance.group(0)):
                                return None
                            else:
                                string = string[road_appearance.span()[1]:]
                        else:
                            # Nếu có các cụm gần phố thì thôi (kiểu gần phố không thôi, không có mấy cái kiểu gần phố A B gì cả)        
                            if re.search('(?:gần|cạnh|cách|\Wra|giáp|sát) (?:phố|mặt phố)', string_lower) or re.search(r'(?:phố|mặt phố)\s(?:\S+\s)?(?:gần|cạnh|cách|ra\W|giáp|sát|vào)', string_lower):
                                return None
                            return 'Mặt phố'                
                else:
                    if re.search('(?:gần|cạnh|cách|\Wra|giáp|sát) (?:phố|mặt phố)', string_lower) or re.search(r'(?:phố|mặt phố)\s(?:\S+\s)?(?:gần|cạnh|cách|ra\W|giáp|sát|vào)', string_lower):
                        return None
                    return 'Mặt phố'
        return None

    @staticmethod
    def extract_street_or_alley_front(row):
        #------TH0: check từ short_address------
        road_add = row['short_address'].split(',')[0].lower()
        if '/' in road_add:
            # Đường 2/9 (đại khái không phải xoẹt)
            if re.search(r'(?:đường|phố) (?:\S+\s){0,2}(?:\S+/\S+)', road_add):
                pass 
            else:
                return 'Mặt ngõ'
        if re.search(r'hẻm|ngõ|ngách|kiệt\s', road_add):
            return 'Mặt ngõ'
        
        if pd.notna(row['description']):
            des = row['description']
        else:
            des = ''
        if pd.notna(row['title']):
            title = row['title']
        else:
            title = ''

        string = title + '. ' + des
        short_add = row['short_address']
        if short_add is None:
            short_add = ''
        pho = DataCleaner.search_pho(string, short_add)
        string = string.lower()
        #------ TH1: Hẻm xe hơi, hẻm xe tải và sẹc ------
        if re.search(r'hxh|hxt|sẹc|sẹt|xẹc|xẹt| sec ', string):
            return 'Mặt ngõ'
        
        #------ TH2: Ngách ------
        result = re.findall(r'(?:\S+\s+){0,5}(\S+\sngách)\s*(?:\S+\s+){0,5}', string)
        if result != []:
            for i in result:
                if i.startswith('ng'):
                    if ratio(i, 'ngóc ngách') >= 90:
                        continue
                    return 'Mặt ngõ'
                return 'Mặt ngõ'
            
        #------ TH3: Kiệt ------
        kiet = re.findall(r'(?:\S+\s){0,3}kiệt(?:\s\S+){0,2}', string)
        if len(kiet) > 0:
            for k in kiet:
                if re.search(r'(?:lý thường |võ văn |phạm |anh |em |mr.\s?|tuấn |tam |nhân |văn )kiệt', k) or 'kiệt tác' in k:
                    continue
                else:
                    if pho:
                        break 
                    return 'Mặt ngõ'
        
        #------ TH4: Hẻm ------
        if re.search(r'(?<!như)(?<!hơn)(?:\S+\s){0,2}(?:hẻm|\Whem\W)',string):
            return 'Mặt ngõ'
        
        #------ TH5: Các trường hợp drop define từ search_pho ------
        if pho == 'Drop':
            return None
        
        #------ TH6: Ngõ ------
        ngo = re.findall(r'(?:(?<!hơn\s)(?<!như\s)(?<!giá\s)(?:\S+\s*){1,3})ngõ', string)
        cua_ngo = re.search(r'(?<!đỗ\s)(?:cửa ngõ|cưa ngõ|một mặt ngõ|1 mặt ngõ|một ngõ|1 ngõ)', string)
        if ngo:
            if "nhà ngõ" in string:
                return 'Mặt ngõ'
            if 'mặt ngõ' in string:
                return 'Mặt ngõ'
            if 'ngõ vào' in string:
                return 'Mặt ngõ'
            if not cua_ngo:
                if pho:
                    if re.search(r'như (?:\S+\s){0,2}ngõ', string):
                        return 'Mặt phố'
                    return 'Mặt ngõ' # Ngõ hết
                return 'Mặt ngõ'
            elif cua_ngo and len(ngo) == 1:
                pass
            else:
                if pho:
                    return 'Drop' # Drop hết các dòng trong phần này vì nó lẫn lộn giữa mặt phố và mặt ngõ
                return 'Mặt ngõ'
        if pho:
            slash = re.search(r'nhà \d+/\D', string)
            if slash and slash.group(0) != '24/24' and slash.group(0) != '24/7':
                return 'Mặt ngõ'
            return 'Mặt phố' # return 'Có mặt phố'
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
            ground_truth = pd.read_excel(INFO_FILE,sheet_name="Sheet1")
            print(f"Loaded ground truth file successfully from: {INFO_FILE}")
        except FileNotFoundError:
            print(f"Ground truth file not found.")
            return df

        ground_truth = ground_truth[['Tỉnh/Thành phố', 'Diện tích (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài']]
        ground_truth.dropna(inplace=True)
        ground_truth = ground_truth[ground_truth[target_col] > 0]

        if ground_truth.empty:
            print("No valid values in ground truth file after cleaning.")
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

            # 3. Look up the estate with closest area
            gt_province_subset['area_diff'] = (gt_province_subset['Diện tích (m2)'] - target_area).abs()
            best_match = gt_province_subset.loc[gt_province_subset['area_diff'].idxmin()]
            gt_width = best_match['Kích thước mặt tiền (m)']
            gt_length = best_match['Kích thước chiều dài']

            # 4. Calculate the width from the ground truth estate's shape ratio 
            shape_ratio = gt_width / gt_length
            imputed_width = (target_area * shape_ratio) ** 0.5
            df_imputed.loc[index, target_col] = round(imputed_width, 2)

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
    def fill_missing_distance_to_the_main_road(df):
        """
        Fill missing values in 'Khoảng cách tới trục đường chính (m)' 
        by computing the shortest path distance from each property to 
        the nearest edge matching the main road name from OpenStreetMap.
        """

        target_col = "Khoảng cách tới trục đường chính (m)"
        df_imputed = df.copy()
        rows_to_impute = df_imputed[df_imputed[target_col].isna()].index

        def has_name(x, name):
            """Helper: check if an OSM edge name matches a given street name."""
            if isinstance(x, list):
                return name in x
            return x == name

        for idx in rows_to_impute:
            row = df_imputed.loc[idx]
            lat, lon = row.get("latitude"), row.get("longitude")
            address = row.get("short_address")

            if pd.isna(lat) or pd.isna(lon) or not address:
                continue

            # Extract street name (matches words after 'đường' or 'phố')
            part = address.split(",")[0]
            match = re.search(
                r'\b(?:đường|phố)\s+[A-Za-zÀ-ỹà-ỹĐđ]+(?:\s+[A-Za-zÀ-ỹà-ỹĐđ]+)*',
                part,
                re.IGNORECASE,
            )
            if not match:
                continue

            street = match.group(0).strip()
            print(f"[Row {idx}] Processing address: {address} | Street: {street}")

            try:
                # Load OSM graph around the property
                G = ox.graph_from_point((lat, lon), dist=500, network_type="all")

                # Project graph to local CRS (UTM) → distances now in meters
                G_proj = ox.project_graph(G)

                # Find nearest node to the property
                orig_node = ox.distance.nearest_nodes(G_proj, lon, lat)

                # Find edges that match the street name
                def find_street_edges(name):
                    return [
                        (u, v, k)
                        for u, v, k, data in G_proj.edges(keys=True, data=True)
                        if data.get("name") and any(
                            has_name(n, name)
                            for n in (
                                [data["name"]] if isinstance(data["name"], str) else data["name"]
                            )
                        )
                    ]

                street_edges = find_street_edges(street)

                # Try alternate naming (e.g. remove or add prefixes)
                if not street_edges:
                    alt = street
                    if street.lower().startswith("đường "):
                        alt = street.replace("đường ", "", 1)
                    elif street.lower().startswith("phố "):
                        alt = street.replace("phố ", "", 1)
                    else:
                        alt = "đường " + street

                    street_edges = find_street_edges(alt)

                if not street_edges:
                    print(f"[Row {idx}] No street match for: {street}")
                    continue

                # Pick the nearest street edge (Euclidean in projected CRS)
                orig_x = G_proj.nodes[orig_node]["x"]
                orig_y = G_proj.nodes[orig_node]["y"]

                edge_centers = [
                    (
                        (G_proj.nodes[u]["x"] + G_proj.nodes[v]["x"]) / 2,
                        (G_proj.nodes[u]["y"] + G_proj.nodes[v]["y"]) / 2,
                        (u, v),
                    )
                    for u, v, k in street_edges
                ]

                edge_x, edge_y, (u, v) = min(
                    edge_centers,
                    key=lambda p: (orig_x - p[0]) ** 2 + (orig_y - p[1]) ** 2,
                )

                dest_node = ox.distance.nearest_nodes(G_proj, edge_x, edge_y)

                # Compute the shortest path distance in meters
                distance = nx.shortest_path_length(G_proj, orig_node, dest_node, weight="length")
                df_imputed.loc[idx, target_col] = distance

                print(f"[Row {idx}] Distance computed: {distance:.1f} m")

            except Exception as e:
                print(f"[Row {idx}] Error computing distance: {e}")
                continue

        return df_imputed


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
        estimated_price = row.get('Giá ước tính')
        land_area = row.get('Diện tích đất (m2)')
        construction_cost_per_sqm = row.get('Đơn giá xây dựng')
        remaining_quality = row.get('Chất lượng còn lại')
        total_floor_area = row.get('Tổng diện tích sàn')

        if pd.isna(estimated_price) or pd.isna(construction_cost_per_sqm) or \
                pd.isna(land_area) or pd.isna(remaining_quality) or \
                pd.isna(total_floor_area):
            return None

        if land_area <= 0 or total_floor_area < 0:
            return None

        building_value = construction_cost_per_sqm * total_floor_area * remaining_quality

        if building_value >= estimated_price:
            return round(estimated_price / land_area, 2)

        land_value = estimated_price - building_value
        land_unit_price = land_value / land_area

        return round(land_unit_price, 2)