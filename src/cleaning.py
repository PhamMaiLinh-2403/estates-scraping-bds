from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import random

import numpy as np
import pandas as pd

from src.config import (
    CONSTRUCTION_COST_MAP,
    DEFAULT_QUALITY,
    FACADE_COUNT_MAP,
    FEATURE_KEYWORDS,
    QUALITY_LEVELS,
    SHAPE_KEYWORDS,
    STREET_PREFIXES,
    NON_STREET_KEYWORDS,
    DETAIL_PREFIXES,
    NEGATION_PATTERNS,
)

__all__ = [
    "DataCleaner",
    "drop_mixed_listings",
]


def drop_mixed_listings(df: pd.DataFrame):
    """
    Removes rows from the DataFrame where 'title' or 'description'
    contain the keyword "thổ cư".
    """
    initial_count = len(df)

    if 'title' not in df.columns or 'description' not in df.columns:
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
    # Must refer to the property itself being on a main road, not just near one
    direct_main_road_patterns = [
        r"(nhà|vị trí|căn nhà|biệt thự|lô đất|đất)\s+(ngay\s+)?(mặt\s+(phố|tiền|đường)|mặt\s+tiền\s+đường|mặt\s+tiền)",
        r"(tọa lạc|nằm|vị trí)\s+(ngay\s+)?trên\s+(mặt\s+(phố|đường|tiền))",
        r"(mặt\s+(phố|tiền|đường))\s+(lớn|chính|kinh\s+doanh)"
    ]

    # Negative patterns: ra mặt phố, gần mặt đường, cách mặt phố 50m, view mặt đường
    near_but_not_on_patterns = [
        r"(cách|ra|gần|view)\s+(mặt\s+(phố|đường|tiền))",
        r"(50|30|100)\s*m\s+(ra|tới|cách)\s+(mặt\s+(phố|đường|tiền))"
    ]

    for pat in near_but_not_on_patterns:
        if re.search(pat, text):
            return False

    for pat in direct_main_road_patterns:
        if re.search(pat, text):
            return True

    return False


class DataCleaner:
    """
    Static collection of cleaning / parsing helpers.
    """

    # ----- Basic helpers -----
    @staticmethod
    def _parse_and_clean_number(text_value: Any) -> Optional[float]:
        """
        Extract the first numeric token from a string and return it as a float.
        """
        if not isinstance(text_value, str):
            return None

        # A more robust regex to capture a sequence of digits, dots, and commas.
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
                # Filter for cases with explicit prefixes "đường", "phố"
                match = re.search(r'\b(đường|phố)\s+[^\d,]+', part.lower())
                if match and len(match.group(0).split()) <= 5:
                    return match.group(0).title().strip()

            first = parts[0]

            # Filter for parts that do NOT start with a digit and do not start with non-street keywords
            if not first[0].isdigit() and not first.lower().startswith(NON_STREET_KEYWORDS) \
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
    def extract_address_detail(row: Dict[str, Any]) -> str:
        """
        Extracts specific address details like house/alley numbers.
        If not available, determine if it's 'Mặt phố' or 'Mặt ngõ'.
        """

        # --- Step 1: Analyze 'short_address' with improved logic ---
        short_address = str(row.get("short_address", ""))

        parts = [p.strip() for p in short_address.split(',') if p.strip()]
        is_street = False
        first_part = parts[0].lower()

        if first_part.startswith(STREET_PREFIXES) and not first_part.startswith(NON_STREET_KEYWORDS):
            is_street = True

        if not is_street:
            # If not street, try to extract detail prefix parts
            detail_parts = []
            for part in parts:
                part_lower = part.lower()
                if part_lower.startswith(DETAIL_PREFIXES) or re.match(r'^\d+[\/\w-]*', part.strip()):
                    detail_parts.append(part)
                else:
                    break  # Stop once we hit a part that is clearly not a detail prefix

            if not detail_parts:
                return ""

            final = ", ".join(detail_parts)

            # Remove street-like patterns inside the result
            # e.g., 'Đường Lê Sát', 'Đường Số 16', etc.
            final = re.sub(r'\b(đường|hẻm|ngõ|tổ|khu phố|phố|số nhà|nhà|mặt tiền)[^,]*', '', final, flags=re.IGNORECASE)

            dp = str(row.get("Đường phố", "")).strip().lower()
            if dp:
                final_lower = final.lower()
                if dp in final_lower:
                    final = final_lower.replace(dp, '')

            # Clean up multiple slashes, spaces, etc.
            final = re.sub(r'\s+', ' ', final)  # Normalize whitespace
            final = re.sub(r'(^[,\s]+|[,\s]+$)', '', final)  # Trim leading/trailing commas or spaces
            final = re.sub(r',\s*,+', ',', final)  # Collapse redundant commas

            return final.title()

        # --- Step 2: Fallback to searching text for "Mặt phố" or "Mặt đường" ---
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if is_on_main_road(text):
            return "Mặt phố"

        # --- Step 3: Default to "Mặt ngõ" if no other information is found ---
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
            price_str = price_str.lower().strip()
            num_part = re.search(r"([\d\.,]+)", price_str)

            if not num_part:
                raise ValueError("No numeric part found in price string.")

            # Use the robust cleaning function
            cleaned_num = DataCleaner._parse_and_clean_number(num_part.group(1))
            if cleaned_num is None:
                raise ValueError("Could not parse number")

            if "tỷ" in price_str:
                value = cleaned_num * 1e9
            elif "triệu" in price_str:
                value = cleaned_num * 1e6
            else:
                value = cleaned_num
            return round(value, 2)

        if not isinstance(main_info_json, str):
            return np.nan
        try:
            for item in json.loads(main_info_json):
                if item.get("title") == "Mức giá":
                    value = item.get("value", "")
                    if isinstance(value, str) and value.lower().strip() != "thỏa thuận":
                        return _convert(value)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
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

        if pd.notna(row.get("description")):
            text = str(row["description"])
            patterns = [r"(?i)(?:diện tích|dt)[:\s]*([\d\.,]+)\s*m[²2]", r"([\d\.,]+)\s*m[²2]"]

            for p in patterns:
                m = re.search(p, text)
                if m:
                    area = DataCleaner._parse_and_clean_number(m.group(1))
                    if pd.notna(area):
                        return area

        return np.nan

    @staticmethod
    def extract_num_floors(row: Dict[str, Any]) -> Optional[int]:
        try:
            other_info = json.loads(row.get("other_info", "{}") or "{}")
            if "Số tầng" in other_info:
                m = re.search(r"\d+", str(other_info["Số tầng"]))
                if m:
                    return int(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            main_info = json.loads(row.get("main_info", "[]") or "[]")
            for item in main_info:
                if item.get("title") == "Số tầng":
                    m = re.search(r"\d+", str(item["value"]))
                    if m:
                        return int(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if not text:
            return None

        word_to_num = {
            "một": 1, "hai": 2, "ba": 3,
            "bốn": 4, "năm": 5, "sáu": 6,
            "bảy": 7, "bẩy": 7, "tám": 8,
            "chín": 9, "mười": 10,
        }
        num_words_pattern = "|".join(word_to_num.keys())
        potential_numbers = {
            int(n)
            for n in re.findall(r"(?:cải\s+tạo\s+lên|xây\s+lên|nâng\s+tầng\s+lên|xin\s+phép\s+xây)\s*(\d+)", text)
        }  # For filtering out potential numbers of a building, since we only care about actual number
        candidate_numbers: List[int] = []

        for num_str in re.findall(rf"(\d+|{num_words_pattern})\s*(?:tầng|lầu|tấm|mê)", text):
            if num_str.isdigit():
                candidate_numbers.append(int(num_str))
            elif num_str in word_to_num:
                candidate_numbers.append(word_to_num[num_str])

        actual = [n for n in candidate_numbers if n not in potential_numbers]

        if "lầu" in text and any(k in text for k in ["trệt", "lửng"]):
            extra = ("trệt" in text) + ("lửng" in text)
            if actual:
                return max(actual) + extra
        if actual:
            return max(actual)
        if not actual and candidate_numbers:
            return None
        if "nhà cấp 4" in text or "nhà trệt" in text:
            return 1
        return None

    # ----- Construction & quality -----
    @staticmethod
    def get_construction_cost(row: Dict[str, Any]) -> Optional[int]:
        def _prop_type() -> str:
            """More careful property type detection."""
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            if re.search(r"\bbán\s+(căn\s+)?biệt\s+thự\b", text) or re.search(r"\bnhà\s+(kiểu\s+)?biệt\s+thự\b",
                                                                              text):
                return "biệt thự"
            if re.search(r"\bnhà\s+cấp\s*4\b", text):
                return "nhà cấp 4"
            return "nhà thường"

        def _has_bsmt() -> bool:
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            return bool(re.search(r"\b(tầng\s+)?hầm\b", text))

        # Primary signals
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

        for quality_val, keywords in QUALITY_LEVELS:
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    return round(quality_val, 2)

        return round(DEFAULT_QUALITY, 2)

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
            m = re.search(r"(?:mặt tiền|chiều rộng|chiều ngang)\s*:?\s*([\d.,m]+)", text.lower())

            if m:
                # Use the class's number parsing utility
                return DataCleaner._parse_and_clean_number(m.group(1))
            return None

        # --- Step 1: Check structured 'other_info' JSON field first ---
        try:
            other_info_json = row.get("other_info", "{}") or "{}"
            val = json.loads(other_info_json).get("Mặt tiền")
            w = DataCleaner._parse_and_clean_number(val)
            if w is not None:
                return w
        except (json.JSONDecodeError, TypeError):
            pass

        # --- Step 2: Check structured 'main_info' JSON field ---
        try:
            main_info_json = row.get("main_info", "[]") or "[]"
            for it in json.loads(main_info_json):
                # Check for area info that often contains dimensions in the 'ext' field
                if it.get("title") == "Diện tích" and it.get("ext"):
                    w = _find(it["ext"])
                    if w is not None:
                        return w
        except (json.JSONDecodeError, TypeError):
            pass

        # --- Step 3: Fallback to searching free-text fields ---
        for field in (row.get("description"), row.get("title")):
            # Convert field to string to handle potential non-string types
            w = _find(str(field))
            if w is not None:
                return w

        # Return None if no width was found in any source
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
            m = re.search(r"(?:dài|chiều\s+dài)\s*:?\s*([\d.,m]+)", text_l)
            if m:
                return DataCleaner._parse_and_clean_number(m.group(1))

            m = re.search(r"(\d+[\.,m]?\d*)\s*(?:m|mét)?\s*[xX*]\s*(\d+[\.,m]?\d*)", text_l)
            if m:
                nums = [DataCleaner._parse_and_clean_number(n) for n in m.groups()]
                if all(n is not None for n in nums):
                    # Here, we assume the larger dimension is the length
                    valid_nums = [n for n in nums if n is not None]
                    return max(valid_nums) if valid_nums else None
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
        widths: List[float] = []

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        # Define alley-related keywords and patterns
        alley_kw = r"(?:ngõ|hẻm|ngách|kiệt|lối\s+vào|đường\s+vào|đường\s+trước\s+nhà)"
        width_kw = r"(?:rộng\s*)?"  # optional "rộng"
        num_pat = r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(m|mét)(?!²|2)"

        # Match patterns like "hẻm rộng 3m", "đường vào: 4m", etc.
        pattern = rf"{alley_kw}[\s:–-]*{width_kw}{num_pat}"

        for match in re.findall(pattern, text):
            num_str = match[0]
            w = DataCleaner._parse_and_clean_number(num_str)
            if w is not None:
                widths.append(w)

        if widths:
            return min(widths)

        # Heuristic fallback based on phrases
        if "nhà mặt phố" in text.lower() or "nhà mặt đường" in text.lower():
            return 0
        if "ô tô tránh" in text.lower():
            return 5.0
        if "xe tải tránh" in text.lower():
            return 10.0
        if "xe máy tránh" in text.lower():
            return 2.5
        if any(k in text.lower() for k in ["ô tô vào", "oto vào", "ô tô đỗ cửa", "oto đỗ cửa", "ô tô đỗ tận nơi"]):
            return 3.5

        return None

    @staticmethod
    def extract_distance_to_main_road(row: Dict[str, Any]) -> Optional[float]:
        def _convert(num_str: str, unit: str) -> Optional[float]:
            cleaned_num = DataCleaner._parse_and_clean_number(num_str)
            if cleaned_num is None:
                return None
            return round(cleaned_num * 1000 if unit.lower() == "km" else cleaned_num, 2)

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        if not text.strip():
            return None

        # 1. Direct check if the property is on a main road
        if is_on_main_road(text):
            return 0.0

        # 2. Match patterns like "cách mặt phố 30m" or "30m tới đường lớn" but avoid irrelevant location names
        road_kw = r"(mặt\s+phố|đường\s+lớn|đường\s+chính|trục\s+chính|đường\s+ô\s*tô|phố|đường)"
        unit_kw = r"(km|mét|m)?"
        dist_cap = r"(\d{1,3}(?:[\.,]\d{1,3})?)\s*" + unit_kw

        # Avoid these false matches
        ignore_kw = r"(quận|phường|thành phố|cửa\s+biển|lăng|chợ|hồ|trường|bệnh viện|chùa|khu\s+vực)"

        patt1 = rf"{road_kw}.*?(?:cách|khoảng|tầm|tới|ra|đến)\s*{dist_cap}"
        patt2 = rf"(?:cách|khoảng|tầm|tới|ra|đến)\s*{dist_cap}\s*(?:đến|tới|ra)?\s*{road_kw}"

        matches = re.findall(patt1, text) + re.findall(patt2, text)
        dists = []

        for match in matches:
            match_text = " ".join(match)
            if re.search(ignore_kw, match_text):  # Skip if near points of interest
                continue
            num, unit = match[0], match[1] or "m"
            converted = _convert(num, unit)
            if converted is not None and 0 < converted < 1000:  # Set a reasonable range
                dists.append(converted)

        if dists:
            return min(dists)

        # 3. Try to infer from common phrases
        if re.search(r"(ngõ\s+nông|ngõ\s+rộng|ngõ\s+thoáng|ngõ\s+gần\s+đường)", text):
            return 10.0
        if re.search(r"(ngõ\s+xe\s+máy|ngõ\s+hẹp)", text):
            return 20.0
        if re.search(r"(trong\s+ngõ|trong\s+hẻm)", text):
            return 25.0

        # 4. No match: return a random fallback
        return float(random.randint(5, 15))

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