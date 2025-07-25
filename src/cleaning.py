from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import random
import unicodedata

import numpy as np
import pandas as pd

from src.config import (
    CONSTRUCTION_COST_MAP,
    DEFAULT_QUALITY,
    FACADE_COUNT_MAP,
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
    "is_on_main_road",
    "parse_and_clean_width"
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
    """
    Determine whether the property is directly on a main road, not just near one.
    """
    text = text.lower()

    # === 1. Negative indicators — property is NEAR but not ON main road
    near_but_not_on_patterns = [
        r"(cách|ra|gần|view|hướng\s+ra|đi\s+ra|đi\s+ra\s+đến)\s+(mặt\s+(phố|đường|tiền))",
        r"(view|hướng\s+ra)\s+(phố|đường|mặt\s+phố)",
        r"\b\d{1,3}\s*m(?:ét)?\s*(tới|ra|cách)\s+(mặt\s+(phố|đường|tiền))",
        r"\bkhoảng\s*\d{1,3}\s*m\s*(đến|ra|tới|cách)\s+(mặt\s+(phố|đường|tiền))",
        r"(gần|kế|bên cạnh)\s+(phố|đường|mặt\s+phố|mặt\s+tiền)",
        r"(cách|ra|gần|view|hướng\s+ra|đi\s+ra|đi\s+ra\s+đến)\s+(phố|đường|tiền)",
        r"(view|hướng\s+ra)\s+(phố|đường)",
        r"\b\d{1,3}\s*m(?:ét)?\s*(đến|tới|ra|cách)\s+(phố|đường|tiền)",
        r"\bkhoảng\s*\d{1,3}\s*m\s*(đến|ra|tới|cách)\s+(phố|đường|tiền)",
        r"(gần|kế|bên cạnh)\s+(phố|đường)",
        r"\b\d{1,3}\s*m\s*(cách|ra|tới|đến)?\s*(phố|đường|tiền)"
    ]

    for pat in near_but_not_on_patterns:
        if re.search(pat, text):
            return False

    # === 2. Positive indicators — property is ON a main road
    direct_main_road_patterns = [
        r"(nhà|biệt thự|căn nhà|lô đất|đất|vị trí|nằm|tọa lạc|căn hộ)?\s*(ngay\s+)?(mặt\s+(phố|tiền|đường)|mặt\s+tiền)",
        r"(nhà|biệt thự|căn nhà|vị trí|nằm|tọa lạc|căn hộ)?\s*(ngay\s+)?trên\s+(phố|đường|đường\s+chính|phố\s+lớn)",
        r"(nằm|tọa lạc)\s+(trên|tại)\s+trục\s+(đường|phố)\s+(chính|lớn)",
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
                match = re.search(r'\b(đường|phố)\s+[\w\s\-()\/]+', part.lower())
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
            return 1

        word_to_num = {
            "một": 1, "hai": 2, "ba": 3,
            "bốn": 4, "năm": 5, "sáu": 6,
            "bảy": 7, "bẩy": 7, "tám": 8,
            "chín": 9, "mười": 10,
        }
        num_words_pattern = "|".join(word_to_num.keys())
        potential_numbers = re.findall(
            r"(?:"
            r"cải\s+tạo\s+lên|"  # cải tạo lên 4
            r"xây\s+lên|"  # xây lên 3
            r"nâng\s+tầng\s+lên|"  # nâng tầng lên 5
            r"xin\s+phép\s+xây|"  # xin phép xây 6
            r"có\s+thể\s+lên|"  # có thể lên 4
            r"có\s+khả\s+năng\s+xây|"  # có khả năng xây 7
            r"móng(?:\s+cứng)?|"  # móng 7 tầng, móng cứng 8 tầng
            r"thiết\s+kế\s+lên\s+tới"  # thiết kế lên tới 5 tầng
            r")\s*(\d+)",
            text
        )        # For filtering out potential numbers of a building, since we only care about actual number
        candidate_numbers: List[int] = []

        for num_str in re.findall(rf"(\d+|{num_words_pattern})\s*(?:tầng|lầu|tấm|mê)", text):
            if num_str.isdigit():
                candidate_numbers.append(int(num_str))
            elif num_str in word_to_num:
                candidate_numbers.append(word_to_num[num_str])

        actual = [n for n in candidate_numbers if n not in potential_numbers]

        if "lầu" in text and any(k in text for k in ["trệt", "lửng", "gác mái"]):
            extra = ("trệt" in text) + ("lửng" in text) + ("gác mái" in text)
            if actual:
                return max(actual) + extra
        if actual:
            return max(actual)
        if not actual and candidate_numbers:
            return None
        if "nhà cấp 4" in text or "nhà trệt" in text:
            return 1
        return 1

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

        # Patterns: "diện tích 4 x 15", "dt: 5.5 x 12m", "DT 6 x 17 m²"
        size_match = re.search(r"(diện\s+tích|dt|DT)[:\s]*([\d.,]+)\s*x\s*([\d.,]+)", desc)
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

            m2 = re.search(r"(?:diện\s+tích|dt)[:\s]*([\d.,]+)\s*x\s*([\d.,]+)", text_l)
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

        # === 0. Explicitly on the main road
        if is_on_main_road(norm_text):
            return 0.0

        try:
            other_info_json = row.get("other_info", "{}") or "{}"
            val = json.loads(other_info_json).get("Đường vào")
            w = parse_and_clean_width(val)
            return w if w and w < 10 else None
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
            ("hẻm thông thoáng", 2.5),
            ("ngõ thông", 2.5),
            ("hẻm thông", 2.5),
            ("đường thông", 2.5),
        ]

        for kw, width in descriptive_fallback:
            if unicodedata.normalize('NFC', kw.lower()) in norm_text:
                return width if width < 15 else None

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
            r"đường\s+[a-z0-9/]+",  # e.g., đường 23/10, đường số 5
            r"phố\s+[a-z0-9/]+",
            r"trục\s+chính", r"đường\s+lớn", r"đường\s+chính",
            r"đường\s+ô\s*tô", r"mặt\s+phố", r"mặt\s+tiền",
            "ô tô đỗ", "chỗ đỗ xe"
        ]
        road_kw = f"(?:{'|'.join(road_prefixes)})"

        # Units and number pattern
        unit_kw = r"(km|m|mét)?"
        dist_cap = r"(\d{1,3}(?:[\.,]\d{1,2})?)\s*" + unit_kw

        # Avoid matching these points of interest
        place_of_interest = r"(bigc|vincom|trường|chợ|bệnh viện|công viên|khu\s+vui\s+chơi|tttm|siêu thị|trung\s+tâm|sân\s+vận\s+động|bến\s+xe|cafe|nhà\s+hàng)"

        # Main patterns
        patt1 = rf"{road_kw}.*?(cách|khoảng|tầm|tới|ra|đến)\s*{dist_cap}"
        patt2 = rf"(cách|khoảng|tầm|tới|ra|đến)\s*{dist_cap}\s*(đến|tới|ra)?\s*{road_kw}"

        matches = re.findall(patt1, text) + re.findall(patt2, text)
        dists = []

        for match in matches:
            match_text = " ".join(str(m) for m in match)
            if re.search(place_of_interest, match_text):  # Skip places like BigC, Vincom...
                continue
            num, unit = match[-2], match[-1] or "m"
            converted = _convert(num, unit)
            if converted is not None and 0 < converted < 1000:
                dists.append(converted)

        if dists:
            return min(dists)

        # === 3. Phrase-based inference
        if re.search(r"(ngõ\s+nông|ngõ\s+rộng|ngõ\s+thoáng|ngõ\s+gần\s+đường)", text):
            return 10.0
        if re.search(r"(ngõ\s+xe\s+máy|ngõ\s+hẹp)", text):
            return 20.0
        if re.search(r"(trong\s+ngõ|trong\s+hẻm)", text):
            return 25.0

        return float(random.randint(10, 200))

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
                pattern=r"(?:diện\s+tích\s+|dt\s+)?sàn(?:\s+\S+){0,5}?\s*:?\s*(\d+[.,]?\d*)\s*(?:m2|m²|m)\b",
                string=des
            )
            if first_pattern:
                result_str = first_pattern.group(1).replace(',', '.')
                try:
                    return float(result_str)
                except ValueError:
                    pass  # Fall through if conversion fails

            # Pattern 2: Look for "<area> m x <floors> T".
            second_pattern = re.search(pattern=r'(\d+[.,]?\d*)\s*m\s*[*|x]\s*(\d+)\s*[T|t]', string=des)
            if second_pattern:
                area_str = second_pattern.group(1).replace(',', '.')
                floor_str = second_pattern.group(2)
                try:
                    area = float(area_str)
                    floor = float(floor_str)
                    return area * floor
                except ValueError:
                    pass  # Fall through

        # # --- Fallback method: Approximate from other columns ---
        # land_area = row.get('Diện tích đất (m2)')
        # num_floors = row.get('Số tầng công trình')

        # # Ensure both values are available and valid before calculating
        # if pd.notna(land_area) and pd.notna(num_floors):
        #     try:
        #         # Check for valid numeric types and that floors > 0
        #         if isinstance(land_area, (int, float)) and isinstance(num_floors, (int, float)) and num_floors > 0:
        #             return round(float(land_area * num_floors), 2)
        #     except (ValueError, TypeError):
        #         # This handles cases where values might be non-numeric strings
        #         pass

        # # If all methods fail, return None.
        return None