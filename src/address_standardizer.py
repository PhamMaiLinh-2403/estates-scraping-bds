import sqlite3
import pandas as pd
from typing import Optional

class AddressStandardizer:
    def __init__(self, provinces_sql_path: str, districts_sql_path: str):
        self.reverse_province_map = {}
        self.reverse_district_map = {}
        self._load_data(provinces_sql_path, districts_sql_path)

    def _load_data(self, provinces_sql_path: str, districts_sql_path: str):
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE provinces (name TEXT, code TEXT, status TEXT);")
            conn.execute("CREATE TABLE districts (name TEXT, code TEXT, province_code TEXT, status TEXT);")

            with open(provinces_sql_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

            with open(districts_sql_path, "r", encoding="utf-8") as f:
                dis_cleaned = f.read().replace("\\'", "''")
                conn.executescript(dis_cleaned)

            provinces_df = pd.read_sql_query("SELECT * FROM provinces", conn)
            districts_df = pd.read_sql_query("SELECT * FROM districts", conn)

        except (FileNotFoundError, sqlite3.Error) as e:
            print(f"Error: Could not load administrative data. Details: {e}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()

        # Build reverse lookup maps
        self.reverse_province_map = {
            prov.replace("Thành phố ", "").replace("Tỉnh ", ""): prov
            for prov in provinces_df['name'].unique()
        }
        self.reverse_province_map['Bà Rịa Vũng Tàu'] = 'Tỉnh Bà Rịa - Vũng Tàu'

        self.reverse_district_map = {
            dis.replace("Thành phố ", "").replace("Thành Phố ", "").replace("Huyện ", "")
               .replace("Quận ", "").replace("Thị xã ", ""): dis
            for dis in districts_df['name'].unique()
        }


    def standardize_province(self, province_name: Optional[str]) -> Optional[str]:
        if not isinstance(province_name, str):
            return province_name
        return self.reverse_province_map.get(province_name, province_name)

    def standardize_district(self, district_name: Optional[str]) -> Optional[str]:
        if not isinstance(district_name, str):
            return district_name
        prefixes = ['Thành phố', 'Thành Phố', 'Quận', 'Huyện', 'Thị xã']
        if any(prefix in district_name for prefix in prefixes):
            return district_name
        if district_name == 'Plei Ku':
            return 'Thành phố Pleiku'
        if district_name == 'Tuy Hòa':
            return 'Thành phố Tuy Hòa'
        if district_name == 'Đảo Phú Quý':
            return 'Thành phố Phan Thiết'
        if district_name == 'Việt Yên':
            return 'Thị xã Việt Yên'
        return self.reverse_district_map.get(district_name, district_name)