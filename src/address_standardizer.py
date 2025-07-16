import sqlite3
import pandas as pd
from typing import Optional

class AddressStandardizer:
    """
    Loads official administrative data from SQL files and provides methods
    to standardize province and district names.
    """
    def __init__(self, provinces_sql_path: str, districts_sql_path: str):
        """
        Initializes the standardizer by loading data from SQL files into memory.
        """
        self.reverse_province_map = {}
        self.reverse_district_map = {}
        self._load_data(provinces_sql_path, districts_sql_path)

    def _load_data(self, provinces_sql_path: str, districts_sql_path: str):
        """Loads data from SQL files and builds reverse lookup maps."""
        try:
            conn = sqlite3.connect(":memory:")

            # Create tables
            conn.execute("CREATE TABLE provinces (name TEXT, code TEXT, status TEXT);")
            conn.execute("CREATE TABLE districts (name TEXT, code TEXT, province_code TEXT, status TEXT);")

            # Execute insert scripts
            with open(provinces_sql_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

            with open(districts_sql_path, "r", encoding="utf-8") as f:
                # Clean script for safe execution (handles escaped quotes)
                dis_cleaned = f.read().replace("\\'", "''")
                conn.executescript(dis_cleaned)

            # Read into pandas DataFrames
            provinces_df = pd.read_sql_query("SELECT * FROM provinces", conn)
            districts_df = pd.read_sql_query("SELECT * FROM districts", conn)

        except FileNotFoundError as e:
            print(f"Error: SQL file not found. Make sure paths are correct in config.py. Details: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

        # Build reverse province map
        self.reverse_province_map = {
            prov.replace("Thành phố ", "").replace("Tỉnh ", ""): prov
            for prov in provinces_df['name'].unique()
        }
        # Add manual corrections
        self.reverse_province_map['Bà Rịa Vũng Tàu'] = 'Tỉnh Bà Rịa - Vũng Tàu'

        # Build reverse district map
        self.reverse_district_map = {
            dis.replace("Thành phố ", "").replace("Thành Phố ", "").replace("Huyện ", "").replace("Quận ", "").replace("Thị xã ", ""): dis
            for dis in districts_df['name'].unique()
        }

    def standardize_province(self, province_name: Optional[str]) -> Optional[str]:
        """
        Standardizes a province name using the loaded map.
        Returns the original name if not found or input is invalid.
        """
        if not isinstance(province_name, str):
            return province_name
        return self.reverse_province_map.get(province_name, province_name)

    def standardize_district(self, district_name: Optional[str]) -> Optional[str]:
        """
        Standardizes a district name using a combination of rules and the loaded map.
        Returns the original name if not found or input is invalid.
        """
        if not isinstance(district_name, str):
            return district_name

        # If it already has a prefix, it's likely already standard.
        prefixes = ['Thành phố', 'Thành Phố', 'Quận', 'Huyện', 'Thị xã']
        if any(prefix in district_name for prefix in prefixes):
            return district_name

        # Manual corrections for ambiguous cases
        if district_name == 'Plei Ku':
            return 'Thành phố Pleiku'
        if district_name == 'Tuy Hòa':
            return 'Thành phố Tuy Hòa'
        if district_name == 'Đảo Phú Quý':
            return 'Thành phố Phan Thiết'  # Note: This seems like a custom rule, keeping as requested
        if district_name == 'Việt Yên':
            return 'Thị xã Việt Yên'

        # Default lookup
        return self.reverse_district_map.get(district_name, district_name)