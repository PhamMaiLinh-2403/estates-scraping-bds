import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

class DatabaseManager:
    """
    Manages SQLite database operations for real estate scraping pipeline.
    Handles raw data, cleaned data, and metadata storage with deduplication.
    """
    
    def __init__(self, db_path: str = "output/real_estate.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create all necessary tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Raw scraped data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    listing_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    short_address TEXT,
                    address_parts TEXT,
                    latitude REAL,
                    longitude REAL,
                    main_info TEXT,
                    description TEXT,
                    other_info TEXT,
                    image_urls TEXT,
                    content_hash TEXT NOT NULL,
                    status TEXT DEFAULT 'NEW',
                    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(url, content_hash)
                )
            """)
            
            # Index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_url 
                ON raw_listings(url)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_raw_listing_id 
                ON raw_listings(listing_id)
            """)
            
            # Table 2: Cleaned data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cleaned_listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_listing_id INTEGER NOT NULL,
                    province TEXT,
                    district TEXT,
                    ward TEXT,
                    street TEXT,
                    address_details TEXT,
                    transaction_status TEXT,
                    transaction_date DATE,
                    contact_info TEXT,
                    price_listed REAL,
                    price_estimated REAL,
                    price_unit_type TEXT,
                    land_unit_price REAL,
                    business_advantage TEXT,
                    num_floors REAL,
                    total_floor_area REAL,
                    construction_cost_per_sqm REAL,
                    construction_year INTEGER,
                    remaining_quality REAL,
                    land_area REAL,
                    facade_width REAL,
                    length REAL,
                    num_facades INTEGER,
                    land_shape TEXT,
                    alley_width REAL,
                    distance_to_main_road REAL,
                    land_use_purpose TEXT,
                    other_factors TEXT,
                    latitude REAL,
                    longitude REAL,
                    image_urls TEXT,
                    cleaned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (raw_listing_id) REFERENCES raw_listings(id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cleaned_raw_id 
                ON cleaned_listings(raw_listing_id)
            """)
            
            # Table 3: Scraping metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraping_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scrape_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    status TEXT NOT NULL,
                    total_urls INTEGER DEFAULT 0,
                    successful_scrapes INTEGER DEFAULT 0,
                    failed_scrapes INTEGER DEFAULT 0,
                    new_records INTEGER DEFAULT 0,
                    changed_records INTEGER DEFAULT 0,
                    duplicate_records INTEGER DEFAULT 0,
                    error_message TEXT,
                    config_snapshot TEXT,
                    worker_id INTEGER
                )
            """)
            
            # Table 4: URL queue for scraping
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS url_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'PENDING',
                    attempts INTEGER DEFAULT 0,
                    last_attempt TIMESTAMP,
                    error_message TEXT,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_url_status 
                ON url_queue(status)
            """)
    
    def _compute_content_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute hash of content fields to detect changes.
        Excludes URL and timestamps from hash.
        """
        hash_fields = [
            str(data.get('title', '')),
            str(data.get('short_address', '')),
            str(data.get('main_info', '')),
            str(data.get('description', '')),
            str(data.get('other_info', '')),
        ]
        
        content = '|'.join(hash_fields)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def insert_raw_listing(self, data: Dict[str, Any]) -> tuple[int, str]:
        """
        Insert or update raw listing data.
        Returns: (record_id, status) where status is 'NEW', 'DUPLICATE', or 'CHANGED'
        """
        content_hash = self._compute_content_hash(data)
        url = data.get('url')
        listing_id = str(data.get('id', '')).replace('.0', '')
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if URL exists with same content
            cursor.execute("""
                SELECT id, content_hash FROM raw_listings 
                WHERE url = ? 
                ORDER BY scraped_at DESC 
                LIMIT 1
            """, (url,))
            
            existing = cursor.fetchone()
            
            if existing:
                if existing['content_hash'] == content_hash:
                    # Exact duplicate
                    return existing['id'], 'DUPLICATE'
                else:
                    # Content has changed
                    status = 'CHANGED'
            else:
                # New record
                status = 'NEW'
            
            # Insert new record
            cursor.execute("""
                INSERT INTO raw_listings (
                    listing_id, url, title, short_address, address_parts,
                    latitude, longitude, main_info, description, 
                    other_info, image_urls, content_hash, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                listing_id,
                url,
                data.get('title'),
                data.get('short_address'),
                data.get('address_parts'),
                data.get('latitude'),
                data.get('longitude'),
                data.get('main_info'),
                data.get('description'),
                data.get('other_info'),
                data.get('image_urls'),
                content_hash,
                status
            ))
            
            return cursor.lastrowid, status
    
    def bulk_insert_raw_listings(self, listings: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk insert raw listings with deduplication.
        Returns counts of new, changed, and duplicate records.
        """
        stats = {'NEW': 0, 'CHANGED': 0, 'DUPLICATE': 0}
        
        for listing in listings:
            _, status = self.insert_raw_listing(listing)
            stats[status] += 1
        
        return stats
    
    def add_urls_to_queue(self, urls: List[str]):
        """Add URLs to scraping queue."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for url in urls:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO url_queue (url) 
                        VALUES (?)
                    """, (url,))
                except sqlite3.IntegrityError:
                    pass  # URL already exists
    
    def get_pending_urls(self, limit: Optional[int] = None) -> List[str]:
        """Get URLs that are pending scraping."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT url FROM url_queue 
                WHERE status = 'PENDING' OR (status = 'FAILED' AND attempts < 3)
                ORDER BY added_at
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            return [row['url'] for row in cursor.fetchall()]
    
    def update_url_status(self, url: str, status: str, error_message: Optional[str] = None):
        """Update URL scraping status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE url_queue 
                SET status = ?, 
                    attempts = attempts + 1,
                    last_attempt = CURRENT_TIMESTAMP,
                    error_message = ?
                WHERE url = ?
            """, (status, error_message, url))
    
    def start_scraping_session(self, scrape_type: str, worker_id: Optional[int] = None, 
                              config_snapshot: Optional[Dict] = None) -> int:
        """Start a new scraping session and return session ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize config snapshot, filtering out non-serializable objects
            config_json = None
            if config_snapshot:
                try:
                    # Filter to only include basic types
                    serializable_config = {
                        k: v for k, v in config_snapshot.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
                    config_json = json.dumps(serializable_config, default=str)
                except (TypeError, ValueError):
                    config_json = None
            
            cursor.execute("""
                INSERT INTO scraping_metadata (
                    scrape_type, start_time, status, worker_id, config_snapshot
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                scrape_type,
                datetime.now(),
                'RUNNING',
                worker_id,
                config_json
            ))
            
            return cursor.lastrowid
    
    def end_scraping_session(self, session_id: int, stats: Dict[str, Any]):
        """Update scraping session with final statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE scraping_metadata 
                SET end_time = ?,
                    status = ?,
                    total_urls = ?,
                    successful_scrapes = ?,
                    failed_scrapes = ?,
                    new_records = ?,
                    changed_records = ?,
                    duplicate_records = ?,
                    error_message = ?
                WHERE id = ?
            """, (
                datetime.now(),
                stats.get('status', 'COMPLETED'),
                stats.get('total_urls', 0),
                stats.get('successful_scrapes', 0),
                stats.get('failed_scrapes', 0),
                stats.get('new_records', 0),
                stats.get('changed_records', 0),
                stats.get('duplicate_records', 0),
                stats.get('error_message'),
                session_id
            ))
    
    def get_raw_listings_for_cleaning(self, limit: Optional[int] = None) -> List[Dict]:
        """Get raw listings that haven't been cleaned yet."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT r.* 
                FROM raw_listings r
                LEFT JOIN cleaned_listings c ON r.id = c.raw_listing_id
                WHERE c.id IS NULL
                ORDER BY r.scraped_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def insert_cleaned_listing(self, raw_listing_id: int, cleaned_data: Dict[str, Any]):
        """Insert cleaned listing data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO cleaned_listings (
                    raw_listing_id, province, district, ward, street,
                    address_details, transaction_status, transaction_date,
                    contact_info, price_listed, price_estimated, price_unit_type,
                    land_unit_price, business_advantage, num_floors, total_floor_area,
                    construction_cost_per_sqm, construction_year, remaining_quality,
                    land_area, facade_width, length, num_facades, land_shape,
                    alley_width, distance_to_main_road, land_use_purpose,
                    other_factors, latitude, longitude, image_urls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                raw_listing_id,
                cleaned_data.get('province'),
                cleaned_data.get('district'),
                cleaned_data.get('ward'),
                cleaned_data.get('street'),
                cleaned_data.get('address_details'),
                cleaned_data.get('transaction_status'),
                cleaned_data.get('transaction_date'),
                cleaned_data.get('contact_info'),
                cleaned_data.get('price_listed'),
                cleaned_data.get('price_estimated'),
                cleaned_data.get('price_unit_type'),
                cleaned_data.get('land_unit_price'),
                cleaned_data.get('business_advantage'),
                cleaned_data.get('num_floors'),
                cleaned_data.get('total_floor_area'),
                cleaned_data.get('construction_cost_per_sqm'),
                cleaned_data.get('construction_year'),
                cleaned_data.get('remaining_quality'),
                cleaned_data.get('land_area'),
                cleaned_data.get('facade_width'),
                cleaned_data.get('length'),
                cleaned_data.get('num_facades'),
                cleaned_data.get('land_shape'),
                cleaned_data.get('alley_width'),
                cleaned_data.get('distance_to_main_road'),
                cleaned_data.get('land_use_purpose'),
                cleaned_data.get('other_factors'),
                cleaned_data.get('latitude'),
                cleaned_data.get('longitude'),
                cleaned_data.get('image_urls')
            ))
    
    def export_to_excel(self, table_name: str, output_path: str):
        """Export a table to Excel file."""
        import pandas as pd
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_excel(output_path, index=False)
            print(f"Exported {len(df)} records to {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Raw listings stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'NEW' THEN 1 ELSE 0 END) as new,
                    SUM(CASE WHEN status = 'CHANGED' THEN 1 ELSE 0 END) as changed,
                    SUM(CASE WHEN status = 'DUPLICATE' THEN 1 ELSE 0 END) as duplicate
                FROM raw_listings
            """)
            stats['raw_listings'] = dict(cursor.fetchone())
            
            # Cleaned listings stats
            cursor.execute("SELECT COUNT(*) as total FROM cleaned_listings")
            stats['cleaned_listings'] = dict(cursor.fetchone())
            
            # URL queue stats
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM url_queue
                GROUP BY status
            """)
            stats['url_queue'] = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Recent scraping sessions
            cursor.execute("""
                SELECT * FROM scraping_metadata 
                ORDER BY start_time DESC 
                LIMIT 10
            """)
            stats['recent_sessions'] = [dict(row) for row in cursor.fetchall()]
            
            return stats