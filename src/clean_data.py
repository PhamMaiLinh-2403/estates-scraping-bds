#!/usr/bin/env python3
"""
Data cleaning script that processes raw listings and stores cleaned data in database.
"""

import json
import pandas as pd
from tqdm import tqdm

from src import config
from src.database_manager import DatabaseManager
from src.address_standardizer import AddressStandardizer
from src.cleaning import DataCleaner, DataImputer, FeatureEngineer

def parse_json_field(value):
    """Safely parse JSON field."""
    if pd.isna(value) or value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value

def clean_single_listing(raw_data: dict, standardizer: AddressStandardizer) -> dict:
    """
    Clean a single raw listing and return cleaned data dictionary.
    """
    # Convert to DataFrame row for compatibility with existing cleaning functions
    row = pd.Series(raw_data)
    
    # Parse JSON fields
    row['address_parts'] = parse_json_field(raw_data.get('address_parts'))
    row['main_info'] = parse_json_field(raw_data.get('main_info'))
    row['other_info'] = parse_json_field(raw_data.get('other_info'))
    row['image_urls'] = parse_json_field(raw_data.get('image_urls'))
    
    # Extract address components
    province = DataCleaner.extract_city(row)
    district = DataCleaner.extract_district(row)
    ward = DataCleaner.extract_ward(row)
    street = DataCleaner.extract_street(row)
    
    # Standardize addresses
    if province:
        province = standardizer.standardize_province(province)
    
    row['Tỉnh/Thành phố'] = province
    row['Thành phố/Quận/Huyện/Thị xã'] = district
    
    if district and province:
        district = standardizer.standardize_district(row)
        row['Thành phố/Quận/Huyện/Thị xã'] = district
    
    row['Xã/Phường/Thị trấn'] = ward
    if ward and district and province:
        ward = standardizer.standardize_ward(row)
        row['Xã/Phường/Thị trấn'] = ward
    
    # Extract other fields
    cleaned_data = {
        'province': province,
        'district': district,
        'ward': ward,
        'street': street,
        'address_details': DataCleaner.extract_address_details(row),
        'transaction_status': 'Đang rao bán',  # Default value, can be enhanced
        'transaction_date': DataCleaner.extract_published_date(raw_data.get('main_info', '{}')),
        'contact_info': None,  # Not currently extracted
        'price_listed': DataCleaner.extract_price(row),
        'price_estimated': None,  # Will be calculated later
        'price_unit_type': 'Đ/m2',  # Default
        'land_unit_price': None,  # Will be calculated later
        'business_advantage': None,  # Will be calculated later
        'num_floors': DataCleaner.extract_num_floors(row),
        'total_floor_area': None,  # Will be calculated later
        'construction_cost_per_sqm': None,  # Will be calculated later
        'construction_year': None,  # Not currently extracted
        'remaining_quality': DataCleaner.estimate_remaining_quality(row),
        'land_area': DataCleaner.extract_total_area(row),
        'facade_width': DataCleaner.extract_width(row),
        'length': DataCleaner.extract_length(row),
        'num_facades': DataCleaner.extract_facade_count(row),
        'land_shape': DataCleaner.extract_land_shape(row),
        'alley_width': DataCleaner.extract_adjacent_lane_width(row),
        'distance_to_main_road': DataCleaner.extract_distance_to_the_main_road(row),
        'land_use_purpose': DataCleaner.extract_land_use(row),
        'other_factors': DataCleaner.extract_street_or_alley_front(row),
        'latitude': raw_data.get('latitude'),
        'longitude': raw_data.get('longitude'),
        'image_urls': raw_data.get('image_urls')
    }
    
    # Update row with extracted data for further calculations
    row['Diện tích đất (m2)'] = cleaned_data['land_area']
    row['Kích thước mặt tiền (m)'] = cleaned_data['facade_width']
    row['Số tầng công trình'] = cleaned_data['num_floors']
    row['Chất lượng còn lại'] = cleaned_data['remaining_quality']
    
    # Calculate derived fields
    cleaned_data['construction_cost_per_sqm'] = DataCleaner.extract_construction_cost(row)
    row['Đơn giá xây dựng'] = cleaned_data['construction_cost_per_sqm']
    
    construction_area = DataCleaner.extract_construction_area(row)
    building_area = DataCleaner.extract_building_area(row)
    
    cleaned_data['total_floor_area'] = building_area
    row['Tổng diện tích sàn'] = building_area
    
    # Fill missing length
    row['Kích thước chiều dài (m)'] = cleaned_data['length']
    cleaned_data['length'] = DataImputer.fill_missing_length(row)
    
    # Calculate price-related fields
    row['Giá rao bán/giao dịch'] = cleaned_data['price_listed']
    cleaned_data['price_estimated'] = FeatureEngineer.calculate_estimated_price(row)
    row['Giá ước tính'] = cleaned_data['price_estimated']
    
    cleaned_data['land_unit_price'] = FeatureEngineer.calculate_land_unit_price(row)
    
    # Calculate business advantage
    row['Khoảng cách tới trục đường chính (m)'] = cleaned_data['distance_to_main_road']
    row['Độ rộng ngõ/ngách nhỏ nhất (m)'] = cleaned_data['alley_width']
    cleaned_data['business_advantage'] = FeatureEngineer.calculate_business_advantage(row)
    
    return cleaned_data

def clean_data_pipeline(db_manager: DatabaseManager, batch_size: int = 100):
    """
    Main data cleaning pipeline.
    Reads raw listings, cleans them, and stores in cleaned_listings table.
    """
    print("=" * 80)
    print("🧹 DATA CLEANING PIPELINE")
    print("=" * 80)
    
    # Initialize address standardizer
    print("📍 Initializing address standardizer...")
    standardizer = AddressStandardizer(
        provinces_sql_path=str(config.PROVINCES_SQL_FILE),
        districts_sql_path=str(config.DISTRICTS_SQL_FILE),
        wards_sql_path=str(config.WARDS_SQL_FILE),
        streets_sql_path=str(config.STREETS_SQL_FILE)
    )
    
    # Start cleaning session
    session_id = db_manager.start_scraping_session(
        scrape_type='DATA_CLEANING',
        config_snapshot=None
    )
    
    stats = {
        'total_urls': 0,
        'successful_scrapes': 0,
        'failed_scrapes': 0,
        'status': 'COMPLETED'
    }
    
    # Process in batches
    while True:
        # Get uncleaned listings
        raw_listings = db_manager.get_raw_listings_for_cleaning(limit=batch_size)
        
        if not raw_listings:
            print("✅ No more listings to clean")
            break
        
        print(f"\n📦 Processing batch of {len(raw_listings)} listings...")
        stats['total_urls'] += len(raw_listings)
        
        # Clean each listing
        for raw_listing in tqdm(raw_listings, desc="Cleaning"):
            try:
                cleaned_data = clean_single_listing(raw_listing, standardizer)
                db_manager.insert_cleaned_listing(raw_listing['id'], cleaned_data)
                stats['successful_scrapes'] += 1
            except Exception as e:
                print(f"\n❌ Error cleaning listing {raw_listing['id']}: {e}")
                stats['failed_scrapes'] += 1
    
    # Update session
    db_manager.end_scraping_session(session_id, stats)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 CLEANING SUMMARY")
    print("=" * 80)
    print(f"Total Processed:   {stats['total_urls']}")
    print(f"Successfully Cleaned: {stats['successful_scrapes']}")
    print(f"Failed:            {stats['failed_scrapes']}")
    print(f"Session ID:        {session_id}")
    print("=" * 80)

def main():
    """Main entry point for cleaning script."""
    db_manager = DatabaseManager(db_path=str(config.OUTPUT_DIR / "real_estate.db"))
    
    print("\n🧹 Real Estate Data Cleaning")
    print("=" * 80)
    
    # Show statistics
    stats = db_manager.get_statistics()
    raw_count = stats['raw_listings']['total']
    cleaned_count = stats['cleaned_listings']['total']
    uncleaned_count = raw_count - cleaned_count
    
    print(f"\nRaw Listings:      {raw_count}")
    print(f"Cleaned Listings:  {cleaned_count}")
    print(f"Pending Cleaning:  {uncleaned_count}")
    
    if uncleaned_count == 0:
        print("\n✅ All listings are already cleaned!")
        return
    
    # Ask for confirmation
    proceed = input(f"\nClean {uncleaned_count} listings? (y/n): ").strip().lower()
    
    if proceed == 'y':
        clean_data_pipeline(db_manager)
        
        # Show final statistics
        stats = db_manager.get_statistics()
        print(f"\n✅ Cleaning complete!")
        print(f"Total cleaned listings: {stats['cleaned_listings']['total']}")
        
        # Ask if user wants to export
        export = input("\nExport cleaned data to Excel? (y/n): ").strip().lower()
        if export == 'y':
            output_path = config.OUTPUT_DIR / "cleaned_listings.xlsx"
            db_manager.export_to_excel('cleaned_listings', str(output_path))
    else:
        print("❌ Cleaning cancelled")

if __name__ == "__main__":
    main()