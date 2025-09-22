import sqlite3
import pandas as pd
from typing import Optional
from unicodedata import normalize

class AddressStandardizer:
    def __init__(self, provinces_sql_path: str, districts_sql_path: str, wards_sql_path: str, streets_sql_path: str):
        self.reverse_province_map = {}
        self.reverse_district_map = {}
        self.provinces_sql_path = provinces_sql_path
        self.districts_sql_path = districts_sql_path
        self.wards_sql_path = wards_sql_path
        self.streets_sql_path = streets_sql_path
        self._load_data()

    def _load_data(self):
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("CREATE TABLE provinces (name TEXT, code TEXT, status TEXT);")
            conn.execute("CREATE TABLE districts (name TEXT, code TEXT, province_code TEXT, status TEXT);")
            conn.execute("CREATE TABLE wards (name TEXT, code TEXT, district_code TEXT, status TEXT);")
            conn.execute("CREATE TABLE streets (name TEXT, code TEXT, district_code TEXT, status TEXT);")

            with open(self.provinces_sql_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())

            with open(self.districts_sql_path, "r", encoding="utf-8") as f:
                dis_cleaned = f.read().replace("\\'", "''")
                conn.executescript(dis_cleaned)

            with open(self.wards_sql_path, "r", encoding="utf-8") as f:
                ward_cleaned = f.read().replace("\\'", "''")
                conn.executescript(ward_cleaned)

            with open(self.streets_sql_path, "r", encoding="utf-8") as f:
                street_cleaned = f.read().replace("'\\'", "''")
                street_cleaned = f.read().replace("'\'", "''")
                conn.executescript(street_cleaned)

            provinces_df = pd.read_sql_query("SELECT * FROM provinces", conn)
            # districts_df = pd.read_sql_query("SELECT * FROM districts", conn)

            # districts_df = pd.read_sql_query("""
            # SELECT d.name AS district_name, d.code AS district_code, d.province_code as province_code, 
            #                                  d.status AS district_status
            # FROM districts d
            # JOIN provinces p ON d.province_code = p.code
            # """, conn)

            # wards_df = pd.read_sql_query("""
            # SELECT w.name AS ward_name, w.code AS ward_code, w.status AS ward_status,
            #        d.name AS district_name, d.code AS district_code,
            #        p.name AS province_name, p.code AS province_code
            # FROM wards w
            # JOIN districts d ON w.district_code = d.code
            # JOIN provinces p ON d.province_code = p.code
            # """, conn)

            # # Join để street có thêm district + province
            # streets_df = pd.read_sql_query("""
            #     SELECT s.name AS street_name, s.code AS street_code, s.status AS street_status,
            #         d.name AS district_name, d.code AS district_code,
            #         p.name AS province_name, p.code AS province_code
            #     FROM streets s
            #     JOIN districts d ON s.district_code = d.code
            #     JOIN provinces p ON d.province_code = p.code
            # """, conn)

            districts_df = pd.read_sql_query("""
                SELECT d.name AS district_name, p.name AS province_name
                FROM districts d
                JOIN provinces p ON d.province_code = p.code
                """, conn)
            
            wards_df = pd.read_sql_query("""
                SELECT w.name AS ward_name,
                    d.name AS district_name,
                    p.name AS province_name
                FROM wards w
                JOIN districts d ON w.district_code = d.code
                JOIN provinces p ON d.province_code = p.code
            """, conn)

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

        self.reverse_district = {}
        for province in districts_df['province_name'].unique():
            self.reverse_district[province] = {}
            for district_name in districts_df[districts_df['province_name'] == province]['district_name'].unique():
                district_name_strip = normalize('NFKD', district_name.replace('Thành phố ', '').replace('Thành Phố ', '').replace('Quận ', '').replace('Huyện ', '').replace('Thị xã ', '').replace('Thị Xã ', '').strip())
                self.reverse_district[province][district_name_strip] = district_name

        self.reverse_ward = {}
        for district in wards_df['district_name'].unique():
            self.reverse_ward[district] = {}
            for ward_name in wards_df[wards_df['district_name'] == district]['ward_name'].unique():
                ward_name_strip = normalize('NFKD', ward_name.replace('Xã ', '').replace('Phường ', '').replace('Thị trấn ', '').replace('Thị Trấn ', '').strip())
                self.reverse_ward[district][ward_name_strip] = ward_name
        # self.reverse_district_map = {
        #     dis.replace("Thành phố ", "").replace("Thành Phố ", "").replace("Huyện ", "")
        #        .replace("Quận ", "").replace("Thị xã ", ""): dis
        #     for dis in districts_df['name'].unique()
        # }
        
        self.districts = districts_df
        self.wards = wards_df
        # self.streets = streets_df

    def standardize_province(self, province_name: Optional[str]) -> Optional[str]:
        if not isinstance(province_name, str):
            return province_name
        return self.reverse_province_map.get(province_name, province_name)

    def standardize_district(self, row) -> Optional[str]:
        if isinstance(row['Thành phố/Quận/Huyện/Thị xã'], str):
            return row['Thành phố/Quận/Huyện/Thị xã']
        short_add_list = row['short_address'].split(',')
        if len(short_add_list) >= 3:
            district_true = normalize('NFKD', short_add_list[-2].strip())
            province = row['Tỉnh/Thành phố']
            if province in self.reverse_district.keys():
                # Add this line to see what the keys actually are
                # print(f"Keys in reverse_district[{province}] are: {list(self.reverse_district[province].keys())}")
                # print(f"The district_true value is: '{district_true}'")
                if self.reverse_district[province][district_true]:
                    # print(f'Return value: {self.reverse_district[province][district_true]}')
                    return self.reverse_district[province][district_true]
            return None
    
    def standardize_ward(self, row):
        if isinstance(row['Xã/Phường/Thị trấn'], str):
            return row['Xã/Phường/Thị trấn']
        short_add_list = row['short_address'].split(',')
        quanhuyen = row['Thành phố/Quận/Huyện/Thị xã']
        ward_names = self.wards[self.wards['district_name'] == quanhuyen]['ward_name'].unique()

        if len(short_add_list) >= 4:
            for ward in ward_names:
                if short_add_list[-3] in ward:
                    return ward
        return None