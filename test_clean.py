import json
import re
from typing import Any, Dict, List, Optional
import unicodedata

import numpy as np
import pandas as pd

# Helper function (originally standalone) required by other extractors
def is_on_main_road(text: str) -> bool:
    """
    Determine whether the property is directly on a main road, not just near one.
    This version is used as a rule, not for fallback.
    """
    if not isinstance(text, str):
        return False
    text = text.lower()

    # Negative indicators — property is NEAR but not ON main road
    near_but_not_on_patterns = [
        r"(cách|ra|gần|view|hướng\s+ra|đi\s+ra|đi\s+ra\s+đến)\s+(mặt\s+(phố|đường|tiền))",
        r"\b\d{1,100}\s*m(?:ét)?\s*(tới|ra|cách)\s+(mặt\s+(phố|đường|tiền))",
        r"(gần|kế|bên cạnh|kề)\s+(phố|đường|mặt\s+phố|mặt\s+tiền)",
        r"(cách|ra|tới|đến)\s+(phố|đường(?:\s+lớn)?|mặt\s+(phố|đường|tiền))\s*\d{1,100}\s*m(?:ét)?"
    ]
    for pat in near_but_not_on_patterns:
        if re.search(pat, text):
            return False

    # Positive indicators — property is ON a main road
    direct_main_road_patterns = [
        r"(nhà|biệt thự|căn nhà|lô đất|đất|đất nền|vị trí|nằm|tọa lạc|căn hộ|ở)?\s*(ngay\s+)?(mặt\s+(phố|tiền|đường)|mặt\s+tiền)",
        r"(nhà|biệt thự|căn nhà|vị trí|nằm|tọa lạc|căn hộ|ở)?\s*(ngay\s+)?trên\s+(phố|đường|đường\s+chính|phố\s+lớn)",
        r"(nằm|tọa lạc|ở)\s+(trên|tại)\s+trục\s+(đường|phố)\s+(chính|lớn)",
    ]
    for pat in direct_main_road_patterns:
        if re.search(pat, text):
            return True
            
    return False

# Helper function (originally standalone) required by other extractors
def parse_and_clean_width(text_value: Any) -> Optional[float]:
    """
    Parses a string that might contain a number and cleans it into a float.
    Handles formats like '1.234,56' -> 1234.56 or '1,234' -> 1234.0
    """
    if not isinstance(text_value, str):
        return None

    match = re.search(r"([\d\.,]+)", text_value)
    if not match:
        return None
    num_str = match.group(1)

    # Standardize number format
    if "," in num_str and "." in num_str:
        cleaned_num_str = num_str.replace(".", "").replace(",", ".")
    elif "," in num_str:
        cleaned_num_str = num_str.replace(",", ".")
    else:
        cleaned_num_str = num_str
        
    try:
        value = float(cleaned_num_str)
        # Handle cases like "5,5m" which becomes 5.5
        # but avoid misinterpreting "5,500" as 5.5
        if value > 20 and "," in num_str and "." not in num_str:
             value = float(num_str.replace(",", ""))
        return round(value, 2)
    except (ValueError, TypeError):
        return None

class RefactoredDataCleaner:
    """
    A collection of refactored cleaning helpers.
    Each function is modified to only use regex and direct rules for extraction,
    removing any imputation or fallback logic. If a value isn't found, it returns None.
    """

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
        cleaned_num_str = num_str.replace(".", "").replace(",", ".")
        try:
            return round(float(cleaned_num_str), 2)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def extract_num_floors(row: Dict[str, Any]) -> Optional[int]:
        """
        Extracts the number of floors using regex and keyword matching.
        Returns None if no specific number is found.
        """
        # --- Constants required for this function ---
        floor_keywords = ["tầng", "lầu", "tấm", "mê"]
        word_to_num = {
            "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6,
            "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9, "mười": 10,
        }
        extra_floor_terms = {"trệt": 1, "lửng": 1, "gác": 1}
        exclude_context = re.compile(r"\d+\s*(m|m2|mét|triệu|tỷ)\b")

        # --- Extraction logic ---
        try: # Check structured JSON fields first
            other_info = json.loads(row.get("other_info", "{}") or "{}")
            if "Số tầng" in other_info:
                m = re.search(r"\d+", str(other_info["Số tầng"]))
                if m: return int(m.group(0))
        except (json.JSONDecodeError, TypeError): pass

        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if not text: return None

        # Pattern for explicit totals like "gồm 5 tầng"
        explicit_total_pattern = re.compile(r"(?:gồm|tổng cộng|có tất cả)\s*(\d+)\s*(?:%s)" % "|".join(floor_keywords))
        m_explicit = explicit_total_pattern.search(text)
        if m_explicit: return int(m_explicit.group(1))

        # Find all numeric or word-based floor mentions
        candidate_numbers: List[int] = []
        num_words_pattern = "|".join(sorted(word_to_num.keys(), key=lambda x: -len(x)))
        floor_pattern = re.compile(rf"(\d+|{num_words_pattern})\s*(?:{'|'.join(floor_keywords)})")
        for num_str in floor_pattern.findall(text):
            if exclude_context.search(num_str): continue
            if num_str.isdigit(): candidate_numbers.append(int(num_str))
            elif num_str in word_to_num: candidate_numbers.append(word_to_num[num_str])

        # Handle composite descriptions like "1 trệt 2 lầu"
        composite_pattern = re.compile(r"(\d+)\s*(trệt|lửng|gác|tầng|lầu|tấm|mê)")
        total_from_composite = 0
        found_composite = False
        for count, term in composite_pattern.findall(text):
            found_composite = True
            total_from_composite += int(count)
        if found_composite: return total_from_composite

        if candidate_numbers: return max(candidate_numbers)
        
        # Check for single-floor keywords if no numbers are found
        if any(x in text for x in ["nhà cấp 4", "nhà trệt"]): return 1

        return None

    @staticmethod
    def extract_facade_width(row: Dict[str, Any]) -> Optional[float]:
        """
        Extracts facade width using regex from various fields.
        Returns None if not found.
        """
        def _find(text: str) -> Optional[float]:
            if not text: return None
            m = re.search(r"(?:mặt tiền|chiều rộng|chiều ngang|rộng|ngang)\s*:?\s*([\d.,]+\s*m?)\b", text.lower())
            if m: return parse_and_clean_width(m.group(1))
            return None

        # Step 1: Check structured 'other_info'
        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Mặt tiền")
            if val and (w := parse_and_clean_width(val)) is not None: return w
        except (json.JSONDecodeError, TypeError): pass

        # Step 2: Search free-text fields
        for field in (row.get("description"), row.get("title")):
            if field and (w := _find(str(field))) is not None: return w

        # Step 3: Extract from "aa x bb m" patterns
        desc = str(row.get("description", "")).lower()
        size_match = re.search(r"(diện\s+tích|dt|kích\s+thước)\s*[:\-]?\s*([\d.,]+)\s*m?\s*[xX*]\s*([\d.,]+)\s*m?", desc)
        if size_match:
            num1 = RefactoredDataCleaner._parse_and_clean_number(size_match.group(2))
            num2 = RefactoredDataCleaner._parse_and_clean_number(size_match.group(3))
            if num1 is not None and num2 is not None: return min(num1, num2)

        return None

    @staticmethod
    def extract_land_length(row: Dict[str, Any]) -> Optional[float]:
        """
        Extracts land length using regex from various fields.
        Returns None if not found.
        """
        def _find(text: str) -> Optional[float]:
            if not text: return None
            text_l = text.lower()
            m = re.search(r"(?:dài|chiều\s+dài)\s*:?\s*([\d.,]+)", text_l)
            if m: return RefactoredDataCleaner._parse_and_clean_number(m.group(1))
            
            m2 = re.search(r"(diện\s+tích|dt|kích\s+thước)\s*[:\-]?\s*([\d.,]+)\s*m?\s*[xX*]\s*([\d.,]+)\s*m?", text_l)
            if m2:
                num1 = RefactoredDataCleaner._parse_and_clean_number(m2.group(2))
                num2 = RefactoredDataCleaner._parse_and_clean_number(m2.group(3))
                if num1 is not None and num2 is not None: return max(num1, num2)
            return None

        # Step 1: Check structured 'other_info'
        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Chiều dài")
            if val and (length := RefactoredDataCleaner._parse_and_clean_number(val)): return length
        except (json.JSONDecodeError, TypeError): pass
        
        # Step 2: Search free-text fields
        for field in (row.get("description"), row.get("title")):
            if field and (length := _find(str(field))): return length
            
        return None

    @staticmethod
    def extract_alley_width(row: Dict[str, Any]) -> Optional[float]:
        """
        Extracts alley width from text using only regex patterns.
        Returns None if no explicit width is mentioned.
        """
        text = f"{row.get('title', '')} {row.get('description', '')}"
        if not text.strip(): return None
        norm_text = unicodedata.normalize('NFC', text.lower())

        # Check structured JSON first
        try:
            val = json.loads(row.get("other_info", "{}") or "{}").get("Đường vào")
            if val and (w := parse_and_clean_width(val)): return w
        except (json.JSONDecodeError, TypeError): pass

        # Regex-based extraction from text
        alley_kw = r"(?:ngõ|hẻm|ngách|kiệt|đường\s+vào|lối\s+vào|trước\s+nhà)"
        num_pat = r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(m|mét)(?!²|2)"
        
        patterns = [
            rf"\b{alley_kw}\b\s*{num_pat}\b",
            rf"{num_pat}\b\s*\b{alley_kw}\b",
            rf"\b{alley_kw}\b[^.,;:\n\r]{{0,20}}\b{num_pat}\b",
        ]

        widths: List[float] = []
        for pattern in patterns:
            for match in re.findall(pattern, norm_text):
                num_str = match[0] if isinstance(match, tuple) else match
                if (width := parse_and_clean_width(num_str)) is not None:
                    widths.append(width)
        
        # Rule: if it's explicitly on the main road, width is not applicable in the same way.
        # Returning 0.0 indicates it's on the main road.
        if is_on_main_road(norm_text):
            return 0.0

        if widths:
            # Filter out illogical values and return the smallest valid one
            valid_widths = [w for w in widths if w < 15]
            if valid_widths:
                return min(valid_widths)

        return None

    @staticmethod
    def extract_distance_to_main_road(row: Dict[str, Any]) -> Optional[float]:
        """
        Extracts the distance to a main road using only regex patterns.
        Returns 0.0 if on the main road, None otherwise if not found.
        """
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if not text.strip(): return None

        if is_on_main_road(text): return 0.0

        # Regex definitions
        road_kw = r"(?:đường\s+[a-z0-9/]+|phố\s+[a-z0-9/]+|trục\s+chính|đường\s+lớn|mặt\s+phố|mặt\s+đường)"
        unit_kw = r"(km|m|mét)?"
        dist_cap = rf"(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*{unit_kw}"
        
        # Patterns to find distances
        patterns = [
            rf"{road_kw}.*?(cách|khoảng)\s*{dist_cap}",
            rf"(?:cách|khoảng)\s*{dist_cap}\s*(?:đến|tới|ra)?\s*{road_kw}",
            rf"{dist_cap}\s*(?:đến|tới|ra|cách)?\s*{road_kw}",
            rf"(?:cách|ra|tới|đến)\s+{road_kw}\s*{dist_cap}"
        ]

        dists = []
        for pat in patterns:
            for match in re.findall(pat, text):
                num, unit = match[-2], match[-1] or "m"
                cleaned_num = RefactoredDataCleaner._parse_and_clean_number(num)
                if cleaned_num is not None:
                    converted = cleaned_num * 1000 if unit.lower() == "km" else cleaned_num
                    if 0 < converted < 1000: # Filter illogical distances
                        dists.append(converted)

        if dists: return min(dists)

        return None


# --- Main execution logic ---
if __name__ == "__main__":
    INPUT_FILE = "output/listing_details.csv"
    OUTPUT_FILE = "cleaned_subset_test.xlsx"

    print(f"--- Starting Cleaning Test ---")
    
    try:
        print(f"1. Reading data from '{INPUT_FILE}'...")
        # Take a subset for faster testing, e.g., first 1000 rows
        df = pd.read_csv(INPUT_FILE).head(1000)
        print(f"   Loaded {len(df)} rows for processing.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please ensure the raw data file is in the same directory as this script.")
        exit()

    # Apply the cleaning functions to each row and create new columns
    print("2. Applying refactored cleaning functions...")

    df['Số tầng công trình (Cleaned)'] = df.apply(
        lambda row: RefactoredDataCleaner.extract_num_floors(row.to_dict()), axis=1
    )
    df['Kích thước mặt tiền (Cleaned)'] = df.apply(
        lambda row: RefactoredDataCleaner.extract_facade_width(row.to_dict()), axis=1
    )
    df['Kích thước chiều dài (Cleaned)'] = df.apply(
        lambda row: RefactoredDataCleaner.extract_land_length(row.to_dict()), axis=1
    )
    df['Độ rộng ngõ/ngách nhỏ nhất (Cleaned)'] = df.apply(
        lambda row: RefactoredDataCleaner.extract_alley_width(row.to_dict()), axis=1
    )
    df['Khoảng cách đến trục đường chính (Cleaned)'] = df.apply(
        lambda row: RefactoredDataCleaner.extract_distance_to_main_road(row.to_dict()), axis=1
    )
    
    print("   Cleaning functions applied successfully.")

    # Save the result to an Excel file
    try:
        print(f"3. Saving results to '{OUTPUT_FILE}'...")
        # Select relevant columns for easier review in Excel
        output_columns = [
            'title',
            'description',
            'main_info',
            'other_info',
            'Số tầng công trình (Cleaned)',
            'Kích thước mặt tiền (Cleaned)',
            'Kích thước chiều dài (Cleaned)',
            'Độ rộng ngõ/ngách nhỏ nhất (Cleaned)',
            'Khoảng cách đến trục đường chính (Cleaned)'
        ]
        
        # Include original columns if they exist
        original_cols_to_include = [col for col in df.columns if col not in output_columns]
        
        final_df = df[output_columns + original_cols_to_include]

        final_df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"   Successfully saved the output.")
    except Exception as e:
        print(f"Error: Could not save the Excel file. Reason: {e}")

    print("--- Test Complete ---")