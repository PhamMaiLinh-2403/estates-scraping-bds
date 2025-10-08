from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import random
import unicodedata
from rapidfuzz import fuzz

import numpy as np
import pandas as pd

from src.config import *

__all__ = [
    "DataCleaner",
    "drop_mixed_listings",
    "is_on_main_road",
    "parse_and_clean_width"
]



def drop_mixed_listings(df: pd.DataFrame):
    """
    Removes rows from the DataFrame where 'title' or 'description'
    contain the keyword "thổ cư".
    """
    initial_count = len(df)

    if 'title' not in df.columns and 'description' not in df.columns:
        return df

    # Create a combined text series for searching, handling NaNs
    text_to_search = df['title'].fillna('') + ' ' + df['description'].fillna('')

    # Create a boolean mask for rows containing "thổ cư" (case-insensitive)
    mask = text_to_search.str.contains('thổ cư', case=False, na=False)

    # Filter the DataFrame to keep only the rows where the mask is False
    df_filtered = df[~mask]

    final_count = len(df_filtered)
    removed_count = initial_count - final_count

    if removed_count > 0:
        print(f"Removed {removed_count} listings containing 'thổ cư'.")

    return df_filtered.reset_index(drop=True)

def is_on_main_road(text: str) -> bool:
    """
    Determine whether the property is directly on a main road, not just near one.
    """
    text = text.lower()

    # === 1. Negative indicators — property is NEAR but not ON main road
    near_but_not_on_patterns = [
        r"(cách|ra|gần|view|hướng\s+ra|đi\s+ra|đi\s+ra\s+đến)\s+(mặt\s+(phố|đường|tiền))",
        r"(view|hướng\s+ra)\s+(phố|đường|mặt\s+phố)",
        r"\b\d{1,100}\s*m(?:ét)?\s*(tới|ra|cách)\s+(mặt\s+(phố|đường|tiền))",
        r"\bkhoảng\s*\d{1,100}\s*m\s*(đến|ra|tới|cách)\s+(mặt\s+(phố|đường|tiền))",
        r"(gần|kế|bên cạnh|kề)\s+(phố|đường|mặt\s+phố|mặt\s+tiền)",
        r"(cách|ra|gần|view|hướng\s+ra|đi\s+ra|đi\s+ra\s+đến)\s+(phố|đường|tiền)",
        r"(view|hướng\s+ra)\s+(phố|đường)",
        r"\b\d{1,100}\s*m(?:ét)?\s*(đến|tới|ra|cách)\s+(phố|đường|tiền)",
        r"\bkhoảng\s*\d{1,100}\s*m\s*(đến|ra|tới|cách)\s+(phố|đường|tiền)",
        r"(gần|kế|bên cạnh)\s+(phố|đường)",
        r"\b\d{1,100}\s*m\s*(cách|ra|tới|đến)?\s*(phố|đường|tiền)",
        r"(cách|ra|tới|đến)\s+(phố|đường(?:\s+lớn)?|mặt\s+(phố|đường|tiền))\s*\d{1,100}\s*m(?:ét)?"
    ]

    for pat in near_but_not_on_patterns:
        if re.search(pat, text):
            return False

    # === 2. Positive indicators — property is ON a main road
    direct_main_road_patterns = [
        r"(nhà|biệt thự|căn nhà|lô đất|đất|đất nền|vị trí|nằm|tọa lạc|căn hộ|ở)?\s*(ngay\s+)?(mặt\s+(phố|tiền|đường)|mặt\s+tiền)",
        r"(nhà|biệt thự|căn nhà|vị trí|nằm|tọa lạc|căn hộ|ở)?\s*(ngay\s+)?trên\s+(phố|đường|đường\s+chính|phố\s+lớn)",
        r"(nằm|tọa lạc|ở)\s+(trên|tại)\s+trục\s+(đường|phố)\s+(chính|lớn)",
        r"(nhà|biệt thự|căn nhà|căn hộ)\s+phố"
    ]

    for pat in direct_main_road_patterns:
        if re.search(pat, text):
            return True

    return False

def parse_and_clean_width(text_value: Any) -> Optional[float]:
    if not isinstance(text_value, str):
        return None

    match = re.search(r"([\d\.,]+)", text_value)
    if not match:
        return None
    num_str = match.group(1)

    if "," in num_str:
        cleaned_num_str = num_str.replace(".", "").replace(",", ".")
    else:
        cleaned_num_str = num_str
    try:
        value = float(cleaned_num_str)

        if value > 20:
            value = float(num_str.replace(",", ""))
        return round(value, 2)
    except (ValueError, TypeError):
        return None
    
def is_land_only(row: Dict[str, Any]) -> bool:
    """
    Detects if a listing is for 'land only' based on simple keywords.
    Returns True if it's land only, otherwise False.
    """
    address_parts = row.get("address_parts", [])
    if isinstance(address_parts, list):
        lowered_address_parts = [str(part).lower() for part in address_parts]
        address_text = " ".join(lowered_address_parts)
    else:
        address_text = str(address_parts).lower()
    text = f"{row.get('title', '')} {row.get('description', '')} {address_text}".lower()
    land_keywords = ["bán đất", "bán lô đất", "bán nền đất", "bán đất nền"]

    # Check if any of the land keywords are in the text
    if any(keyword in text for keyword in land_keywords):
        # Add a check to prevent cases like "bán đất tặng nhà"
        if "tặng nhà" in text or "có nhà" in text:
            return False
        return True
        
    return False

def check_ham(text):
    # text = text.replace(',', ' ').replace('+', ' ')
    string = re.findall(pattern=r'(?:(?!được xây|giấy phép xây dựng|gpxd|cải tạo)\b\w+\b\W+){1,7}hầm(?:(?!\schui)\W+\b\w+\b){1,7}', string=text)
    if string:
        for substr in string:
            ham = re.search(pattern=r'tầng|lầu|tấm|mê|\d+|xe|kết cấu|kc|ô tô|thang máy|trệt|lửng', string=substr)
            if ham:
                return True
    return False

def extract_construction_cost(row):
    title = row['title']
    title_lower = title.lower()
    if pd.notna(row['description']):
        des_lower = row['description'].lower()
        text = f'{title_lower} {des_lower}'
    else:
        text = title_lower
        des_lower = 'none'
    text = text.replace(',',' ').replace('+',' ').replace('\n',' ').replace('*', ' ')
    text = ' '.join(text.split())
    des_lower = des_lower.replace(',',' ').replace('+',' ').replace('\n',' ').replace('*', ' ')
    des_lower = ' '.join(des_lower.split())
    if 'nhà trệt' in text:
        if not re.search(r'nhà trệt\s*(?:\S+\s+){0,2}(?:\d*\s*)(?:tầng|lầu|tấm|mê)', text):
            return 4000000
    cap4 = re.search(pattern = r'nhà cấp 4|nhà c4|cấp 4\W|nc4|nhà trệt|nhà nát', string=text)
    if cap4:
        return 4000000 #4,000,000
    if row['Số tầng công trình'] == 1:
        return 6275876
    
    ham = check_ham(text)
    if re.search(r'biệt thự', title_lower) or re.search(r'villa\W', title_lower):
    # if 'biệt thự' in title_lower or 'villa' in title_lower:
        if ham:
            return 12848184
        return 10510920
    if des_lower != 'none' and (re.search(r'biệt thự', des_lower) or re.search(r'villa\W', des_lower)):
        villa_pattern = [
            r'(?:thiết kế|xây)*\s*(?:\S+\s+){0,5} (?:phong cách|kiểu|dạng|kiến trúc|cấu trúc)\s*(?:\S+\s+){0,2} (?:biệt thự|villa\W)', #Các nhà có cấu trúc villa
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
            if re.search(pattern, des_lower):
                if ham:
                    return 12848184 # Biệt thự có hầm
                return 10510920 # Biệt thự không hầm
        for pattern in not_villa_pattern:
            if re.search(pattern, des_lower):
                if row['Số tầng công trình'] == 2:
                    if ham:
                        return 6275876 # Nhà 1 tầng 1 hầm
                    else:
                        return 8221171 # Nhà 2 tầng không hầm
                if row['Số tầng công trình'] < 2:
                    if ham:
                        return 6275876 # ví dụ như nhà 1.5 thì 0.5 đó chính là tầng hầm 
                    return 8221171 # Nhà 2 tầng không hầm
                if ham:
                    return 9504604 # Nhà hơn 2 tầng, có hầm
                return 8221171 # Nhà hơn 2 tầng, không hầm
        if ham:
            return 12848184 # Biệt thự có hầm
        return 10510920 # Biệt thự không hầm
    if row['Số tầng công trình'] == 2:
        if ham:
            return 6275876 # Nhà 1 tầng 1 hầm
        else:
            return 8221171 # Nhà 2 tầng không hầm
    if row['Số tầng công trình'] < 2:
        if ham:
            return 6275876 # ví dụ như nhà 1.5 thì 0.5 đó chính là tầng hầm 
        return 8221171 # Nhà 2 tầng không hầm
    if ham:
        return 9504604 # Nhà hơn 2 tầng, có hầm
    return 8221171 # Nhà hơn 2 tầng, không hầm

class DataCleaner:
    """
    Static collection of cleaning / parsing helpers.
    """

    # ----- Validators -----
    @staticmethod
    def validate_and_format_street_name(street_name: Optional[str]) -> Optional[str]:
        """
        Validates, formats, and cleans a street name according to specific rules.
        """
        if not street_name or not isinstance(street_name, str):
            return None

        name = street_name.strip()

        # Rule: Must not start with a special character (non-alphanumeric).
        if re.match(r'^[^\w]', name):
            return None

        # Rule: Invalidate descriptive names like "đường 12m", "đường rộng", "đường ô tô".
        # We check for a number followed by 'm' or common descriptive words.
        descriptive_pattern = re.compile(
            r'\d+\s*(m(2|²)|mét|m)?\b|'     
            r'\b(rộng|lớn|to|hẹp)\b|'  
            r'\b(tỷ|tầng|đẹp|nhỉnh|đắt|-|vip|ẩm thực)\b|'
            r'\b(bê\s*tông|đất|đá|nhựa|kim loại)\b|'
            r'\b(ô\s*tô|oto)\b',
            re.IGNORECASE
        )
        if descriptive_pattern.search(name):
            return None

        # Rule: If it doesn't have a standard prefix, add "Đường".
        name_lower = name.lower()
        if not name_lower.startswith(('đường ', 'phố ')):
            name = "Đường " + name

        return name

    # ----- Basic helpers -----
    @staticmethod
    def _parse_and_clean_number(text_value: Any) -> Optional[float]:
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
        
    # ----- Location extractors -----
    @staticmethod
    def extract_city(row: Dict[str, Any]) -> Optional[str]:
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
    def extract_district(row: Dict[str, Any]) -> Optional[str]:
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
    def extract_ward(row: Dict[str, Any]) -> Optional[str]:
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
    def extract_street(row: Dict[str, Any]) -> Optional[str]:
        short_address = str(row.get("short_address", "")).strip()
        if short_address:
            parts = [p.strip() for p in short_address.split(",")]
            for part in parts:
                match = re.search(r'\b(đường|phố)\s+[\w\s\-()\/]+', part.lower())
                if match and len(match.group(0).split()) <= 5:
                    return match.group(0).title().strip()

            first = parts[0]

            # Filter for parts that do NOT start with a digit and do not start with non-street keywords
            if first and not first[0].isdigit() and not first.lower().startswith(NON_STREET_KEYWORDS) \
                    and any(c.isalpha() for c in first) and len(first.split()) <= 5:
                return "Đường " + first.title()

        parts_raw = str(row.get("address_parts", "")).strip()
        if parts_raw:
            try:
                addr_list = json.loads(parts_raw)
                if addr_list:
                    last_item = addr_list[-1]
                    m = re.search(r"tại (đường|phố)\s+([^,]+)", last_item, re.IGNORECASE)
                    if m and len(m.group(0).split()) <= 5:
                        return f"{m.group(1).capitalize()} {m.group(2).strip()}"
            except (json.JSONDecodeError, ValueError, SyntaxError):
                pass

        title = str(row.get("title", "")).strip()
        if title:
            m = re.search(r"(đường|phố)\s+([\w\s\d\-]+?)(?:,|$|\s-|\s--|\()", title, re.IGNORECASE)
            if m and len(m.group(0).split()) <= 5:
                street_cap = " ".join(w.capitalize() for w in m.group(2).strip().split())
                return f"{m.group(1).capitalize()} {street_cap}"

        return None

    @staticmethod
    def extract_address_detail(row: Dict[str, str]) -> str:
        """
        Extracts specific address details like house/alley numbers.
        If not available, determine if it's 'Mặt phố' or 'Mặt ngõ'.
        """
        short_address = str(row.get("short_address", "")).strip()

        parts = [p.strip() for p in short_address.split(',') if p.strip()]
        detail_parts = []
        is_street = False

        if parts:
            first_part = parts[0].lower()
            if first_part.startswith(STREET_PREFIXES) and not first_part.startswith(NON_STREET_KEYWORDS):
                is_street = True

        if not is_street:
            for part in parts:
                part_lower = part.lower()
                if part_lower.startswith(DETAIL_PREFIXES) or re.match(r'^\d+[\/\w-]*', part.strip()):
                    detail_parts.append(part)
                else:
                    break

            final = ", ".join(detail_parts)

            # Remove embedded street-related terms
            final = re.sub(r'\b(đường|phố|số nhà|hẻm|ngõ|kiệt|ngách|mặt tiền)[^,]*', '', final, flags=re.IGNORECASE)

            # Remove Đường phố info if already in detail
            dp = str(row.get("Đường phố", "")).strip().lower()
            if dp and dp in final.lower():
                final = re.sub(re.escape(dp), '', final, flags=re.IGNORECASE)

            # Clean formatting
            final = re.sub(r'\s+', ' ', final)
            final = re.sub(r'(^[,\s]+|[,\s]+$)', '', final)
            final = re.sub(r',\s*,+', ',', final)

            if final:
                return final.title()

        # Fallback 1: Check if the text suggests main road
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if is_on_main_road(text):
            return "Mặt phố"

        # Fallback 2: Default
        return "Mặt ngõ"

    # ----- Datetime & pricing -----
    @staticmethod
    def extract_published_date(main_info_json: str) -> Optional[str]:
        try:
            for item in json.loads(main_info_json):
                if item.get("title") == "Ngày đăng":
                    return item.get("value")
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def extract_total_price(main_info_json: str) -> float:
        def _convert(price_str: str) -> float:
            num_part = re.search(r"([\d\.,]+)", price_str)
            if num_part: 
                cleaned_num = DataCleaner._parse_and_clean_number(num_part.group(1))
                if cleaned_num is None:
                    return price_str

            if "tỷ" in price_str:
                value = cleaned_num * 1e9
            elif "triệu" in price_str:
                value = cleaned_num * 1e6
            elif "nghìn" in price_str:
                value = cleaned_num * 1e3
            else:
                value = cleaned_num
            return round(value, 2)
        
        if not isinstance(main_info_json, str):
            return np.nan
        
        try:
            data = json.loads(main_info_json)

            # Case 1: Dict format
            if isinstance(data, dict):
                price = data.get("Mức giá")
                if price and ("/m²" in price or "/m2" in price):
                    return np.nan
                if isinstance(price, str) and ("thỏa thuận" in price.lower() or "liên hệ" in price.lower()):
                    return np.nan  
                return _convert(price) 

            # Case 2: List of dicts
            elif isinstance(data, list):
                for item in data:
                    if item.get("title") == "Mức giá":
                        ext_price = item.get("ext")
                        value_price = item.get("value")

                        if any(isinstance(text, str) and term in text.lower()
                            for term in ["thỏa thuận", "liên hệ"]
                            for text in [ext_price, value_price]):
                            return np.nan

                        if isinstance(ext_price, str) and not ("/m²" in ext_price or "/m2" in ext_price):
                            return _convert(ext_price)

                        if isinstance(value_price, str) and not ("/m²" in value_price or "/m2" in value_price):
                            return _convert(value_price)

            # If no usable price found
            return np.nan

        except json.JSONDecodeError:
            return np.nan
        except Exception as e:
            print(f"Error parsing price: {e}")
            return np.nan

    # ----- Physical dimensions & counts (facade, area, floors, …) -----
    @staticmethod
    def extract_total_area(row: Dict[str, Any]) -> float:
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
    def extract_num_floors(row):
        def new_check_additional_floor(string, additional_floor):
            result = 0
            for word in additional_floor:
                if word in string:
                    result += 1
            search_st = re.search(pattern=r'(\Wst\W)', string=string)
            if search_st:
                result += 1
            search_gl = re.search(pattern=r'gác lửng|gác lững|ghác lửng|ghác lững', string=string)
            if search_gl:
                result -= 1
            return result 

        def is_float(num):
            try:
                float(num)
                return True
            except:
                return False

        def extract_separate(lower_value, floor_keywords):
            value_list = lower_value.split()
            forbidden_pattern = r'giấy phép xây dựng|giấy phép xây|phép xây dựng|gpxd|có thể xây|cải tạo|được phép xây'
            additional_floor = ['sân thượng', 'sân thương', 'trêt', 'trệt', 'tret', 'tum', 'hầm', 'hâm', 'gác', 'gac', 'lửng', 'lững', 'lừng']
            if 'trệt' in floor_keywords:
                additional_floor = ['sân thượng', 'sân thương',  'tum', 'hầm', 'hâm', 'gác', 'gac', 'lửng', 'lững', 'lừng']
            word_to_num = {
                    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6,
                    "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10,
                    "mười một": 11, "mười hai": 12, "mười ba": 13, "mười bốn": 14,
                    "mười lăm": 15, "mười sáu": 16
                }
            for keyword in floor_keywords: # Check cho từng chiếc keyword
                if keyword in lower_value: # Nếu trong string có một trong những chiếc keyword
                    i = 0
                    while i < len(value_list):
                        # for value in value_list: # Tìm trong list string mà đã được tách ra sẵn
                    #     if keyword in value: # Nếu chiếc keyword đã tìm thấy ban nãy là của từ này
                            # word_index = value_list.index(value) # Lấy index của từ
                        if keyword in value_list[i]: # Tìm index của chiếc từ keyword floor 
                            # Trong trường hợp mà nó bị dính chữ vào với nhau
                            extracted_floor = value_list[i].replace(keyword, '').replace(',','.')
                            if is_float(extracted_floor) or (extracted_floor in word_to_num.keys()): 
                                word_lower = i - 4 if i - 4 >= 0 else 0
                                word_upper = i + 4 if i + 4 < len(value_list) else len(value_list) - 1
                                search_range = ' '.join(value_list[word_lower:word_upper + 1]) # Tìm trong khoảng 5 từ trước - sau của từ
                                forbidden_word = re.search(pattern=forbidden_pattern, string=search_range)
                                if forbidden_word: # Nếu xuất hiện forbidden word
                                    i += 1
                                else:
                                    # if 'hiện trạng' in ' '.join(value_list[word_lower:i]): # Nếu hiện trạng 3 tầng --> return luôn
                                    #     if is_float(extracted_floor):
                                    #         return abs(float(extracted_floor))
                                    #     if extracted_floor in word_to_num.keys():
                                    #         return word_to_num[extracted_floor]
                                    # else:
                                    if is_float(extracted_floor) or (extracted_floor in word_to_num.keys()):
                                        add_key = new_check_additional_floor(search_range, additional_floor)
                                        if is_float(extracted_floor):
                                            return abs(float(extracted_floor)) + add_key
                                        return word_to_num[extracted_floor] + add_key
                                    else:
                                        add_key = new_check_additional_floor(search_range, additional_floor)
                                        if add_key > 0:
                                            return add_key + 1
                                        else:
                                            i += 1
                            # Nếu có thể extract vị trí ở đằng trước keyword đã cho
                            elif i - 1 >= 0:
                                extracted_floor = value_list[i - 1].replace(',', '.')
                                word_lower = i - 4 if i - 4 >= 0 else 0
                                word_upper = i + 4 if i + 4 < len(value_list) else len(value_list) - 1
                                search_range = ' '.join(value_list[word_lower:word_upper + 1]) # Tìm trong khoảng 5 từ trước - sau của từ
                                forbidden_word = re.search(pattern=forbidden_pattern, string=search_range)
                                if forbidden_word: # Nếu xuất hiện forbidden word
                                    i += 1
                                else:
                                    if 'hiện trạng' in ' '.join(value_list[word_lower:i]): # Nếu hiện trạng 3 tầng --> return luôn
                                        if is_float(extracted_floor):
                                            return abs(float(extracted_floor))
                                        elif extracted_floor in word_to_num.keys():
                                            return word_to_num[extracted_floor]
                                        else:
                                            i += 1
                                    else:
                                        if is_float(extracted_floor) or (extracted_floor in word_to_num.keys()):
                                            add_key = new_check_additional_floor(search_range, additional_floor)
                                            if is_float(extracted_floor):
                                                return abs(float(extracted_floor)) + add_key
                                            return word_to_num[extracted_floor] + add_key
                                        else:
                                            add_key = new_check_additional_floor(search_range, additional_floor)
                                            if add_key > 0:
                                                return add_key + 1
                                            else:
                                                i += 1
                            else:
                                i += 1
                        else:
                            i += 1
            
            return None
        
        floor_keywords = ['tầng', 'lầu', 'tấm', 'mê']
        word_to_num = {
                "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6,
                "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10,
                "mười một": 11, "mười hai": 12, "mười ba": 13, "mười bốn": 14,
                "mười lăm": 15, "mười sáu": 16
            }
        # ------- TH1: Thông tin đã có sẵn ở other_info ------
        if row['other_info'] != {} and row['other_info'].get('Số tầng'):
            return int(row['other_info'].get('Số tầng').split()[0])
        # ------ TH2: Nhà cũ/nhà cấp 4 ở title/description ------
        # Xét của description trước do description thường được viết đầy đủ hơn
        if pd.notna(row['description']):
            lower_des = row['description'].lower().replace('+', ' ').replace('x',' ').replace('*', ' ')
            old_house = re.search(pattern=r'nhà (?:\w+\s*){0,5}cũ|bán(?:\s+\S\s*){0,5}đất', string=lower_des)
            if old_house:
                return 0
            cap4 = re.search(pattern = r'nhà cấp 4|nhà c4|cấp 4|nc4|nhà trệt|nhà nát', string=lower_des)
            if cap4:
                if cap4.group(0) == 'nhà trệt':
                    extract_result = extract_separate(lower_des, floor_keywords)
                    if extract_result:
                        return extract_result
                return 1
        if pd.notna(row['title']):
            lower_title = row['title'].lower().replace('+', ' ').replace('x',' ').replace('*', ' ')
            old_house = re.search(pattern=r'nhà (?:\w+\s*){0,5}cũ|bán(?:\s+\S\s*){0,5}đất', string=lower_title)
            if old_house:
                return 0
            cap4 = re.search(pattern = r'nhà cấp 4|nhà c4|cấp 4|nc4|nhà trệt|nhà nát', string=lower_title)
            if cap4:
                if cap4.group(0) == 'nhà trệt':
                    extract_result = extract_separate(lower_title, floor_keywords)
                    if extract_result:
                        return extract_result
                return 1
        # ------ TH3: Xét tổng số tầng ------
            extract_result = extract_separate(lower_title, floor_keywords)
            if extract_result:
                return extract_result
        if pd.notna(row['description']):
            # Trong trường hợp nêu rõ tầng 1, tầng 2,... thì max sẽ là tổng số tầng
            total_pattern = re.findall(pattern=r'(?:tầng|lầu|tấm|mê)\s* ([\d\w]+):', string=lower_des)
            if total_pattern:
                total_floor_num = []
                for digit in total_pattern:
                    if is_float(digit):
                        total_floor_num.append(abs(float(digit)))
                    elif digit in word_to_num.keys():
                        total_floor_num.append(word_to_num[digit])
                if total_floor_num:
                    return max(total_floor_num)
        #------TH4: Xét số tầng mà có miêu tả cấu trúc cụ thể (Kiểu như trệt 2 lầu)------
            else:
                extract_result = extract_separate(lower_des, floor_keywords)
                if extract_result:
                    return extract_result
                else:
                    extract_result = extract_separate(lower_des, ['trệt', 'trêt', 'tret'])
                    if extract_result:
                        return extract_result  
                    extract_result = extract_separate(lower_title, ['trệt', 'trêt', 'tret'])     
                    if extract_result:
                        return extract_result      
        return 1    
    # def extract_num_floors(row: Dict[str, Any]) -> Optional[int]:
    #     floor_keywords = ["tầng", "lầu", "tấm", "mê"]

    #     # Number words mapping
    #     word_to_num = {
    #         "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6,
    #         "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10,
    #         "mười một": 11, "mười hai": 12, "mười ba": 13, "mười bốn": 14,
    #         "mười lăm": 15, "mười sáu": 16
    #     }

    #     # Extra partial floor components
    #     extra_floor_terms = {
    #         "trệt": 1,
    #         "lửng": 1,
    #         "gác": 1,
    #         "gác lửng": 1,
    #         "mái": 1,  # optional: only if counting roof as usable
    #     }

    #     # Patterns to ignore unrelated numbers
    #     exclude_context = re.compile(r"\d+\s*(m|m2|mét|triệu|tỷ)\b")

    #     # Try `other_info`
    #     try:
    #         other_info = json.loads(row.get("other_info", "{}") or "{}") # Extract other_info if it exists
    #         if other_info != {} and "Số tầng" in other_info.keys():
    #         #     num_floor = other_info['Số tầng']
    #         #     if num_floor != '' and num_floor is not None:
    #         #         num_floor = num_floor.split()[0]
    #         #         return int(num_floor)
    #             m = re.search(r"\d+", str(other_info["Số tầng"]))
    #             if m:
    #                 return int(m.group(0))
    #     except (json.JSONDecodeError, TypeError):
    #         pass

    #     # # Try `main_info`
    #     # try:
    #     #     main_info = json.loads(row.get("main_info", "[]") or "[]")
    #     #     for item in main_info:
    #     #         if item.get("title") == "Số tầng":
    #     #             m = re.search(r"\d+", str(item["value"]))
    #     #             if m:
    #     #                 return int(m.group(0))
    #     # except (json.JSONDecodeError, TypeError):
    #     #     pass

    #     # Search in text
    #     text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    #     if not text:
    #         return 1

    #     # Numbers that are potential only (not actual)
    #     potential_numbers = []
    #     potential_pattern = re.compile(
    #         r"(?:cải\s+tạo\s+lên|xây\s+lên|nâng\s+tầng\s+lên|"
    #         r"xin\s+phép\s+xây|có\s+thể\s+lên|có\s+khả\s+năng\s+xây|"
    #         r"móng(?:\s+cứng)?|thiết\s+kế\s+lên\s+tới|"
    #         r"xây\s+tối\s+đa|dự\s+kiến\s+xây|xây\s+được|phép\s+xây\s+tối\s+đa)"
    #         r"\s*(\d+)"
    #     )
    #     for m in potential_pattern.findall(text):
    #         potential_numbers.append(int(m))

    #     # Match "Gồm 5 tầng", "Tổng cộng 7 tầng"
    #     explicit_total_pattern = re.compile(r"(?:gồm|tổng cộng|có tất cả)\s*(\d+)\s*(?:%s)" % "|".join(floor_keywords))
    #     m_explicit = explicit_total_pattern.search(text)
        
    #     if m_explicit:
    #         return int(m_explicit.group(1))

    #     # Match numeric or word floors, skipping excluded contexts
    #     candidate_numbers: List[int] = []
    #     num_words_pattern = "|".join(sorted(word_to_num.keys(), key=lambda x: -len(x)))
    #     floor_pattern = re.compile(rf"(\d+|{num_words_pattern})\s*(?:{'|'.join(floor_keywords)})")

    #     for num_str in floor_pattern.findall(text):
    #         if exclude_context.search(num_str):
    #             continue
    #         if num_str.isdigit():
    #             candidate_numbers.append(int(num_str))
    #         elif num_str in word_to_num:
    #             candidate_numbers.append(word_to_num[num_str])

    #     # Handle composite floor descriptions: "1 trệt 2 lầu"
    #     # composite_pattern = re.compile(r"(\d+)\s*(trệt|lửng|gác|gác lửng|mái|tầng|lầu|tấm|mê)")
    #     composite_pattern = re.compile(rf"(\d+|{num_words_pattern})\s*(trệt|lửng|gác|gác lửng|mái|tầng|lầu|tấm|mê)")
    #     total_from_composite = 0
    #     found_composite = False

    #     for count, term in composite_pattern.findall(text):
    #         found_composite = True

    #         # Addition
    #         if count.isdigit():
    #             num_val = int(count)
    #         else:
    #             num_val = word_to_num.get(count)
    #         # End addition

    #         # num_val = int(count)
    #         # if term in extra_floor_terms or any(term == fk for fk in floor_keywords):
    #         if term in extra_floor_terms or term in floor_keywords:
    #             total_from_composite += num_val

    #     if found_composite and total_from_composite > 0:
    #         return total_from_composite

    #     # Remove potentials from actuals
    #     actual = [n for n in candidate_numbers if n not in potential_numbers]

    #     # Adjust for "lầu + trệt/lửng/gác"
    #     if any(extra in text for extra in extra_floor_terms.keys()):
    #         extra_count = sum(extra in text for extra in extra_floor_terms.keys())
    #         if actual:
    #             return max(actual) + extra_count

    #     # Return best guess
    #     if actual:
    #         return max(actual)
    #     if not actual and candidate_numbers:
    #         return None
    #     if any(x in text for x in ["nhà cấp 4", "nhà trệt"]):
    #         return 1
    #     return 1


    # ----- Construction & quality -----
    @staticmethod
    def get_construction_cost(row: Dict[str, Any]) -> Optional[int]:
        def _prop_type() -> str:
            """More careful property type detection."""
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            if (
                    re.search(r"\bbán\s+(căn\s+)?biệt\s+thự\b", text) or
                    re.search(r"\bnhà\s+(kiểu\s+)?biệt\s+thự\b", text) or
                    re.search(r"\b(căn\s+)?biệt\s+thự\s+(đơn\s+lập|sân\s+vườn|vườn|cao\s+cấp|đẹp|view|4\s+mặt)\b", text)
            ):
                if not re.search(r"\b(liền\s+kề|gần|cách|khu|đối\s+diện|thuộc|gần\s+khu)\s+biệt\s+thự\b", text):
                    return "biệt thự"
            if re.search(r"\bnhà\s+cấp\s*4\b", text):
                return "nhà cấp 4"
            return "nhà thường"

        def _has_bsmt() -> bool:
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            return bool(re.search(r"\b(tầng\s+)?hầm\b", text))
        
        if row.get('is_land', False):
            return 0
        
        # Primary signals0
        ptype = _prop_type()
        floors = DataCleaner.extract_num_floors(row)
        area = DataCleaner.extract_total_area(row)
        has_bsmt = _has_bsmt()

        # If it's clearly a biệt thự or cấp 4, trust it
        if ptype == "biệt thự":
            return CONSTRUCTION_COST_MAP["biệt_thự_có_hầm" if has_bsmt else "biệt_thự"]
        if ptype == "nhà cấp 4":
            return CONSTRUCTION_COST_MAP["nhà_cấp_4"]

        # --- Fallback logic ---
        # Use area + floors to infer structure if ptype is unclear
        if floors is None:
            # Single floor fallback
            return CONSTRUCTION_COST_MAP["nhà_1_tầng_btct"]

        if floors == 1:
            if area is not None and area < 35:
                # Possibly nhà cấp 4 dù không nói rõ
                return CONSTRUCTION_COST_MAP["nhà_cấp_4"]
            return CONSTRUCTION_COST_MAP["nhà_1_tầng_btct"]

        if floors >= 2:
            key = "nhà_gte_2_tầng_có_hầm" if has_bsmt else "nhà_gte_2_tầng_không_hầm"
            return CONSTRUCTION_COST_MAP[key]

        # Final catch-all
        return CONSTRUCTION_COST_MAP["nhà_1_tầng_btct"]

    @staticmethod
    def estimate_remaining_quality(row: Dict[str, Any]) -> float:
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        result = {}

        for quality_val, keywords in QUALITY_LEVELS:
            for kw in keywords:
                pattern = kw.replace(' ', '(?:\s*\w+\s*){0,2} ')
                pattern = pattern.strip()
                pattern = '\W' + pattern + '\W'
                qual = re.search(pattern, text)

                if qual:
                    ratio = fuzz.ratio(qual[0], kw)
                    if quality_val == 0 or quality_val == 1:
                        ratio += 3
                    result[quality_val] = [qual.group(0), ratio]
        
        if result:
            best_quality, (match, score) = max(result.items(), key=lambda x: (x[1][1], x[0]))
            return best_quality
        else:
            return DEFAULT_QUALITY

        # for quality_val, keywords in QUALITY_LEVELS:
        #     for kw in keywords:
        #         if re.search(rf"\b{re.escape(kw)}\b", text):
        #             return round(quality_val, 2)

        # return round(DEFAULT_QUALITY, 2)

    # ----- Land morphology & frontage -----
    @staticmethod
    def extract_land_shape(row: Dict[str, Any]) -> str:
        def is_negated(text: str, kw: str) -> bool:
            for pattern in NEGATION_PATTERNS:
                if re.search(pattern.format(re.escape(kw)), text):
                    return True
            return False

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        for shape, kws in SHAPE_KEYWORDS.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", text) and not is_negated(text, kw):
                    return shape

        return "Chữ nhật"

    @staticmethod
    def extract_facade_width(row: Dict[str, Any]) -> Optional[float]:
        def _find(text: str) -> Optional[float]:
            """Helper to find facade width in a given text string using regex."""
            if not text:
                return None

            # Regex to find keywords for width followed by a number
            m = re.search(r"(?:mặt tiền|chiều rộng|chiều ngang|rộng|ngang)\s*:?\s*([\d.,]+\s*m)\b", text.lower())

            if m:
                return parse_and_clean_width(m.group(1))
            return None

        # --- Step 1: Check structured 'other_info' JSON field first ---
        try:
            other_info_json = row.get("other_info", "{}") or "{}"
            val = json.loads(other_info_json).get("Mặt tiền")
            w = parse_and_clean_width(val)
            if w is not None:
                return w
        except (json.JSONDecodeError, TypeError):
            pass

        # --- Step 2: Check structured 'main_info' JSON field ---
        try:
            main_info_json = row.get("main_info", "[]") or "[]"
            for it in json.loads(main_info_json):
                if it.get("title") == "Diện tích" and it.get("ext"):
                    w = _find(it["ext"])
                    if w is not None:
                        return w
        except (json.JSONDecodeError, TypeError):
            pass

        # --- Step 3: Search common free-text fields (description/title) ---
        for field in (row.get("description"), row.get("title")):
            w = _find(str(field))
            if w is not None:
                return w

        # --- Step 4: Extract from "aa x bb m" patterns in description ---
        desc = str(row.get("description", "")).lower()
        size_match = re.search(r"(diện\s+tích|dt|kích\s+thước)\s*[:\-]?\s*([\d.,]+)\s*m?(?:²)?\s*[xX*]\s*([\d.,]+)\s*m?(?:²)?", desc)

        if size_match:
            num1 = DataCleaner._parse_and_clean_number(size_match.group(2))
            num2 = DataCleaner._parse_and_clean_number(size_match.group(3))
            if num1 is not None and num2 is not None:
                return min(num1, num2)

        return None

    @staticmethod
    def extract_facade_count(row: Dict[str, Any]) -> int:
        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Số mặt tiền")

            if val:
                m = re.search(r"\d+", str(val))
                if m:
                    return int(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        for pattern, val in FACADE_COUNT_MAP:
            if re.search(pattern, text):
                return val
        return 1

    @staticmethod
    def extract_land_length(row: Dict[str, Any]) -> Optional[float]:
        def _find(text: str) -> Optional[float]:
            if not text:
                return None

            text_l = text.lower()

            m = re.search(r"(?:dài|chiều\s+dài)\s*:?\s*([\d.,]+)", text_l)
            if m:
                return DataCleaner._parse_and_clean_number(m.group(1))

            m2 = re.search(r"(diện\s+tích|dt|kích\s+thước)\s*[:\-]?\s*([\d.,]+)\s*m?(?:²)?\s*[xX*]\s*([\d.,]+)\s*m?(?:²)?", text_l)
            if m2:
                num1 = DataCleaner._parse_and_clean_number(m2.group(1))
                num2 = DataCleaner._parse_and_clean_number(m2.group(2))
                if num1 is not None and num2 is not None:
                    return max(num1, num2)

            return None

        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Chiều dài")
            length = DataCleaner._parse_and_clean_number(val)
            if length:
                return length
        except (json.JSONDecodeError, TypeError):
            pass

        for field in (row.get("description"), row.get("title")):
            length = _find(str(field))
            if length:
                return length
        return None

    @staticmethod
    def extract_alley_width(row: Dict[str, Any]) -> Optional[float]:
        """
        Extract alley width from listing title and description.
        """
        text = f"{row.get('title', '')} {row.get('description', '')}"
        norm_text = unicodedata.normalize('NFC', text.lower())

        try:
            other_info_json = row.get("other_info", "{}") or "{}"
            val = json.loads(other_info_json).get("Đường vào")
            w = parse_and_clean_width(val)
            if w:
                return w
            # return w if w and w < 15 else None
        except (json.JSONDecodeError, TypeError):
            pass

        alley_kw = r"(?:ngõ|hẻm|ngách|kiệt|đường\s+vào|lối\s+vào|trước\s+nhà|đường\s+trước\s+nhà)"
        vehicle_kw = r"(?:oto|ô\s*tô|xe\s+hơi|xe\s+tải|ba\s+gác|xe\s+máy)"
        approx_kw = r"(?:rộng\s*)?(?:khoảng|gần|trên\s+dưới|tầm|xấp\s+xỉ)?\s*"
        num_pat = r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(m|mét)(?!²|2)"
        alley_phrase = rf"(?:{alley_kw}(?:\s+{alley_kw})?)"

        widths: List[float] = []

        patterns = [
            rf"\b{alley_phrase}\b\s*{num_pat}\b",
            rf"{num_pat}\b\s*\b{alley_phrase}\b",
            rf"\b{alley_phrase}\b[^.,;:\n\r]{{0,20}}\b{approx_kw}{num_pat}\b",
            rf"\b{approx_kw}{num_pat}\b[^.,;:\n\r]{{0,20}}\b{alley_phrase}\b",
            rf"\b{alley_phrase}\b[^.,;:\n\r]{{0,30}}\b{vehicle_kw}\b[^.,;:\n\r]{{0,20}}\b{approx_kw}{num_pat}\b",
            rf"\b{approx_kw}{num_pat}\b[^.,;:\n\r]{{0,20}}\b{vehicle_kw}\b[^.,;:\n\r]{{0,30}}\b{alley_phrase}\b",
            rf"\btiếp giáp\b[^.,;:\n\r]{{0,20}}\b{alley_phrase}\b[^.,;:\n\r]{{0,20}}\b{approx_kw}{num_pat}\b",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, norm_text):
                if isinstance(match, tuple):
                    num_str = next((m for m in match if re.match(r"^\d", m)), None)
                else:
                    num_str = match
                width = parse_and_clean_width(num_str)
                if width:
                    widths.append(width)

        if widths:
            return min(widths) if min(widths) < 10 else None

        # === 2. Infer from vehicle clues
        vehicle_fallback = [
            ("ngõ xe tải tránh", 10.0),
            ("ngõ xe tải", 8.0),
            ("ngõ ô tô tránh", 5.0),
            ("ngõ oto tránh", 5.0),
            ("ô tô đỗ cửa", 3.5),
            ("oto vào", 3.5),
            ("ô tô vào", 3.5),
            ("ô tô ra", 3.5),
            ("oto đỗ cửa", 3.5),
            ("ô tô quay đầu", 5.0),
            ("hẻm ô tô", 4.0),
            ("hẻm xe hơi", 4.0),
            ("ngõ ô tô", 4.0),
            ("ba gác tránh", 2.5),
            ("xe máy tránh", 2.5),
        ]

        for kw, width in vehicle_fallback:
            if unicodedata.normalize('NFC', kw.lower()) in norm_text:
                return width if width < 15 else None

        # === 3. Descriptive fallback
        descriptive_fallback = [
            ("ngõ thông", 5),
            ("hẻm thông", 3),
            ("ngõ hẹp", 2.5),
            ("hẻm hẹp", 1.5),
        ]

        for kw, width in descriptive_fallback:
            if unicodedata.normalize('NFC', kw.lower()) in norm_text:
                return width if width < 15 else None
            
        # === 0. Explicitly on the main road
        if is_on_main_road(norm_text):
            return 0.0

        return None

    @staticmethod
    def extract_distance_to_main_road(row: Dict[str, Any]) -> Optional[float]:
        """
        Extracts the distance to a main road from text.
        If no specific distance is found, returns a random float between 10.0 and 200.0.
        """

        def _convert(num_str: str, unit: str) -> Optional[float]:
            cleaned_num = DataCleaner._parse_and_clean_number(num_str)
            if cleaned_num is None:
                return None
            return round(cleaned_num * 1000 if unit.lower() == "km" else cleaned_num, 2)

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if not text.strip():
            return None

        # === 1. If it's directly on a main road
        if is_on_main_road(text):
            return 0.0

        # === 2. Regex Patterns
        # Allow these road keywords
        road_prefixes = [
            r"đường\s+[a-z0-9/]+",  
            r"phố\s+[a-z0-9/]+",
            r"trục\s+chính", r"đường\s+lớn", r"đường\s+chính",
            r"đường\s+ô\s*tô", r"mặt\s+phố", r"mặt\s+tiền", r"mặt\s+đường",
            "ô tô đỗ", "chỗ đỗ xe", "ô tô đậu"
        ]
        road_kw = f"(?:{'|'.join(road_prefixes)})"

        # Units and number pattern
        unit_kw = r"(km|m|mét)?"
        dist_cap = r"(\d{1,3}(?:[\.,]\d{1,2})?)\s*" + unit_kw

        # Avoid matching these points of interest
        place_of_interest = r"(bigc|vincom|trường|chợ|bệnh viện|công viên|khu\s+vui\s+chơi|tttm|siêu thị|trung\s+tâm|sân\s+vận\s+động|bến\s+xe|cafe|nhà\s+hàng)"

        # Main patterns
        patt1 = rf"{road_kw}.*?(cách|khoảng|tầm|tới|ra|đến)\s*{dist_cap}"
        patt2 = rf".*(?:cách|khoảng|tầm)\s*{dist_cap}\s*(đến|tới|ra)?\s*{road_kw}"
        patt3 = rf"{dist_cap}\s*(?:đến|tới|ra|cách)?\s*{road_kw}"
        patt4 = rf"(?:cách|ra|tới|đến)\s+{road_kw}\s*{dist_cap}"

        matches = re.findall(patt1, text) + re.findall(patt2, text) + re.findall(patt3, text) + re.findall(patt4, text)
        dists = []

        for match in matches:
            match_text = " ".join(str(m) for m in match)
            if re.search(place_of_interest, match_text):  
                continue
            num, unit = match[-2], match[-1] or "m"
            converted = _convert(num, unit)
            if converted is not None and 0 < converted < 1000:
                dists.append(converted)

        if dists:
            return min(dists)

        # === 3. Phrase-based inference
        if re.search(
            r"(ngõ\s+nông|ngõ\s+rộng|ngõ\s+thoáng|ngõ\s+gần\s+(đường|phố|mặt\s+(phố|đường))|hẻm\s+gần\s+(đường|phố|mặt\s+(phố|đường)))",
            text):
            return 10.0
        if re.search(
            r"(ngõ\s+hẹp|hẻm\s+hẹp|ngõ\s+xe\s+máy(?:\s+vào)?|hẻm\s+xe\s+máy(?:\s+vào)?)",
            text):
            return 20.0
        if re.search(r"(trong\s+hẻm\s+sâu|hẻm\s+sâu)", text):
            return 30.0

        return int(random.randint(10, 200))

    @staticmethod
    def extract_direct_features(row: Dict[str, Any]) -> List[str]:
        direct = row.get('description')
        if not direct or not isinstance(direct, str):
            return []

        direct_list = direct.split('\n')
        new_version = []
        for part in direct_list:
            part_lower = part.lower()
            part_lower_words = part_lower.split()

            if 'liên hệ' in part_lower or 'lh' in part_lower:
                continue
            elif '***' in part_lower:
                continue
            elif ('tỷ' in part_lower_words or
                  'tr' in part_lower_words or
                  ('triệu' in part_lower_words and
                   'thương lượng' not in part_lower and
                   'tốt' not in part_lower_words)):
                continue
            elif ('giá' in part_lower_words and
                  'thương lượng' not in part_lower and
                  'tốt' not in part_lower_words):
                continue
            elif 'xem nhà' in part_lower:
                continue
            else:
                new_version.append(part)
        return new_version

    def extract_built_area(row: Dict[str, Any]) -> Optional[float]:
        """
        Attempts to extract the total built area (tổng diện tích sàn) from the description.
        If not found, it falls back to approximating from land area and floors.
        """
        # --- Primary method: Extract from text ---
        description = row.get('description')
        if isinstance(description, str):
            des = description.lower()

            # Pattern 1: Look for "diện tích sàn" or "dt sàn" followed by a number.
            first_pattern = re.search(
                pattern=(
                    r"(?:diện\s+tích\s+(?:sàn|xây\s+dựng)|dt\s+(?:sàn|xây\s+dựng))"
                    r"(?:\s+\S+){0,5}?\s*:?\s*(\d+[.,]?\d*)\s*(?:m2|m²|m)\b"
                ),
                string=des
            )
            if first_pattern:
                result_str = first_pattern.group(1).replace(',', '.')
                try:
                    return float(result_str)
                except ValueError:
                    pass  

            # Pattern 2: Look for "<area> m x <floors> T".
            second_pattern = re.search(
                pattern=r'(\d+[.,]?\d*)\s*m\s*[*x]\s*(\d+)\s*(?:[Tt]|tầng|tang)\b',
                string=des
            )
            if second_pattern:
                area_str = second_pattern.group(1).replace(',', '.')
                floor_str = second_pattern.group(2)
                try:
                    area = float(area_str)
                    floor = float(floor_str)
                    return area * floor
                except ValueError:
                    pass  
        return None