from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

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
    DETAIL_PREFIXES
)

__all__ = [
    "DataCleaner",
]


class DataCleaner:
    """
    Static collection of cleaning / parsing helpers.
    """

    # ----- Basic helpers -----
    @staticmethod
    def _parse_and_clean_number(text_value: Any) -> Optional[float]:
        """
        Extract the first numeric token from text_value and return as float.
        Handles decimal commas and trailing m / m2 units.
        """
        if not isinstance(text_value, str):
            return None

        match = re.search(r"(\d+[\.,]?\d*)", text_value)
        if not match:
            return None

        num_str = match.group(1).replace(",", ".")
        try:
            return float(num_str)
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
                    return match.group(1).strip()
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
                    return match.group(0).strip().title()
                # if part.lower().startswith(("đường ", "phố ")) and len(part.split()) <= 5:
                #     return part

            first = parts[0]
            non_prefixes = (
                "phường", "xã", "dự án", "quận",
                "huyện", "thị trấn", "số", "thôn",
                "xóm", 'hẻm', 'kiệt'
            )
            # Filter for cases with no explicit prefix
            if not first.lower().startswith(non_prefixes) and any(c.isalpha() for c in first) and len(first.split()) <= 5:
                return first

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
        if short_address:
            parts = [p.strip() for p in short_address.split(',')]

            if parts:
                first_part_lower = parts[0].lower()
                is_street = first_part_lower.startswith(STREET_PREFIXES)

                if not is_street:
                    is_detail_prefix = first_part_lower.startswith(DETAIL_PREFIXES)
                    has_digit = any(char.isdigit() for char in first_part_lower)

                    if is_detail_prefix or has_digit:
                        detail_parts = []
                        for part in parts:
                            part_lower = part.lower()
                            if part_lower.startswith(DETAIL_PREFIXES) or any(char.isdigit() for char in part):
                                detail_parts.append(part)
                            else:
                                break

                        if detail_parts:
                            final = ", ".join(detail_parts).lower()
                            dp = str(row.get('Đường phố', '')).lower()
                            if dp:
                                if dp in final:
                                    return final.replace(dp, '').title()
                                return final
                            return final.title()
                            #return ', '.join(detail_parts)

        # --- Step 2: Fallback to searching text for "Mặt phố" or "Mặt đường" ---
        text_to_search = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if re.search(r'\b(mặt\s+phố|mặt\s+đường)\b', text_to_search):
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
            price_str = price_str.lower().replace(",", ".").strip()
            if "tỷ" in price_str:
                return float(price_str.replace("tỷ", "").strip()) * 1e9
            if "triệu" in price_str:
                return float(price_str.replace("triệu", "").strip()) * 1e6
            return float(price_str)

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
        } # For filtering out potential numbers of a building, since we only care about actual number
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
        # Helper lambdas mirror original
        def _prop_type() -> str:
            text = f"{row.get('title', '')} {row.get('description', '')}".lower()
            if "biệt thự" in text:
                return "biệt thự"
            if "nhà cấp 4" in text:
                return "nhà cấp 4"
            return "nhà thường"

        def _has_bsmt() -> bool:
            return bool(re.search(r"\bhầm\b", f"{row.get('title', '')} {row.get('description', '')}".lower()))

        ptype = _prop_type()
        floors = DataCleaner.extract_num_floors(row)
        has_bsmt = _has_bsmt()

        if ptype == "biệt thự":
            return CONSTRUCTION_COST_MAP["biệt_thự_có_hầm" if has_bsmt else "biệt_thự"]
        if ptype == "nhà cấp 4":
            return CONSTRUCTION_COST_MAP["nhà_cấp_4"]
        if floors is None or floors == 1:
            return CONSTRUCTION_COST_MAP["nhà_1_tầng_btct"]
        if floors >= 2:
            key = "nhà_gte_2_tầng_có_hầm" if has_bsmt else "nhà_gte_2_tầng_không_hầm"
            return CONSTRUCTION_COST_MAP[key]
        return None

    @staticmethod
    def estimate_remaining_quality(row: Dict[str, Any]) -> float:
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        for quality_val, keywords in QUALITY_LEVELS:
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    return quality_val

        return DEFAULT_QUALITY

    # ----- Land morphology & frontage -----
    @staticmethod
    def extract_land_shape(row: Dict[str, Any]) -> str:
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        for shape, kws in SHAPE_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in kws):
                return shape

        return "Chữ nhật"

    @staticmethod
    def extract_facade_width(row: Dict[str, Any]) -> Optional[float]:
        def _find(text: str) -> Optional[float]:
            if not text:
                return None

            m = re.search(r"(?:mặt tiền|mt|ngang|rộng|chiều ngang)\s*:?\s*([\d.,m]+)", text.lower())

            if m:
                return DataCleaner._parse_and_clean_number(m.group(1))
            return None

        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Mặt tiền")
            w = DataCleaner._parse_and_clean_number(val)
            if w is not None:
                return w
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            for it in json.loads(row.get("main_info", "[]") or "[]"):
                if it.get("title") == "Diện tích" and it.get("ext"):
                    w = _find(it["ext"])
                    if w is not None:
                        return w
        except (json.JSONDecodeError, TypeError):
            pass

        for field in (row.get("description"), row.get("title")):
            w = _find(str(field))
            if w is not None:
                return w
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
                    return max(nums)
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

        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Đường vào")
            w = DataCleaner._parse_and_clean_number(val)
            if w is not None:
                widths.append(w)
        except (json.JSONDecodeError, TypeError):
            pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        for num_str in re.findall(r"(?:ngõ|hẻm|ngách|kiệt|đường\s+vào|đường\s+trước\s+nhà)\s*:?\s*([\d.,m]+)", text):
            w = DataCleaner._parse_and_clean_number(num_str)
            if w is not None:
                widths.append(w)

        if widths:
            return min(widths)
        if "ô tô tránh" in text:
            return 5.0
        if "xe tải tránh" in text:
            return 10.0
        if any(k in text for k in ["ô tô vào", "ô tô đỗ cửa", "oto vào"]):
            return 3.5
        return None

    @staticmethod
    def extract_distance_to_main_road(row: Dict[str, Any]) -> Optional[float]:
        def _convert(num: str, unit: str) -> Optional[float]:
            try:
                val = float(num.replace(",", "."))
                return val * 1000 if unit.lower() == "km" else val
            except ValueError:
                return None

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()

        if not text:
            return None

        road_kw = r"(?:mặt\s+phố|đường\s+lớn|đường\s+chính|trục\s+chính|đường\s+ô\s*tô|phố)"
        dist_cap = r"(\d+[\.,]?\d*)\s*(m|mét|km)?"
        patt1 = rf"{road_kw}\s*(?:cách|khoảng)\s*{dist_cap}"
        patt2 = rf"{dist_cap}\s*(?:ra|tới|cách)\s*{road_kw}"
        dists = [
            _convert(num, unit or "m")
            for num, unit in re.findall(patt1, text) + re.findall(patt2, text)
            if _convert(num, unit or "m") is not None
        ]
        return min(dists) if dists else None

    @staticmethod
    def extract_direct_features(row: Dict[str, Any]) -> List[str]:
        found: set[str] = set()
        text = f"{row.get('title', '')}. {row.get('description', '')}".lower()

        if not text:
            return []
        sentences = filter(None, re.split(r"[.\n!?]+", text))

        for sent in sentences:
            for kw in FEATURE_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", sent):
                    found.add(sent.strip())
                    break
        return list(found)