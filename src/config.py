SELENIUM_CONFIG = {
    "headless": False,
    "uc_driver": True,
    "window_size": "1920,1080",
    "args": [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-extensions",
        "--disable-infobars",
        "--disable-web-security",
        "--disable-background-networking",
        "--disable-default-apps",
        "--disable-sync",
        "--disable-translate",
        "--metrics-recording-only",
        "--mute-audio",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu"
    ]
}

# Concurrency settings
MAX_WORKERS = 3

# Scraping details configuration
SCRAPING_DETAILS_CONFIG = {
    "append_mode": True,
    "start_index": 0,
    "count": 3000,
    "stagger_mode": "random",
    "stagger_step_sec": 3.0,
    "stagger_max_sec": 3.0,
}

# Scraping configuration
BASE_URL = "https://batdongsan.com.vn"
SEARCH_PAGE_URL = f"{BASE_URL}//ban-dat"
PAGE_NUMBER = 1  # Starting page number for scraping

# File path settings
OUTPUT_DIR = "output"
URLS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_urls.csv"
DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details.csv"
CLEANED_DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details_cleaned.xlsx"

# Administrative data paths
ADMIN_DATA_DIR = "Dữ liệu địa giới hành chính"
INFO_DATA_DIR = "Dữ liệu thông tin kỹ thuật tài sản"
PROVINCES_SQL_FILE = f"{ADMIN_DATA_DIR}/provinces_20250225_2.sql"
DISTRICTS_SQL_FILE = f"{ADMIN_DATA_DIR}/districts_20250225_2.sql"
WARDS_SQL_FILE = f"{ADMIN_DATA_DIR}/wards_20250225_2.sql"
STREETS_SQL_FILE = f"{ADMIN_DATA_DIR}/streets_20250225_2.sql"
INFO_FILE = f"{INFO_DATA_DIR}/Dữ liệu thông tin kỹ thuật tài sản.xlsx"

# -- Word maps for cleaning -- 
FACADE_COUNT_MAP = [
        # Keywords for 3 facades
        (r'\b3\s+mặt\s+tiền\b', 3),
        (r'\bba\s+mặt\s+tiền\b', 3),

        # Keywords for 2 facades (most common variant after 1)
        (r'\blô\s+góc\s+2\s+mặt\s+tiền\b', 2),
        (r'\b2\s+mặt\s+tiền\b', 2),
        (r'\bhai\s+mặt\s+tiền\b', 2),
        (r'\blô\s+góc\b', 2),  
        (r'\bcăn\s+góc\b', 2),

        # Keywords explicitly stating 1 facade (checked last)
        (r'\b1\s+mặt\s+tiền\b', 1),
        (r'\bmột\s+mặt\s+tiền\b', 1),
    ]

SHAPE_KEYWORDS = {
    # Specific, highly desirable/undesirable features are checked first.
    'Nở hậu': ['nở hậu', 'hậu nở', 'đuôi nở', 'phía sau nở'],
    'Thóp hậu': ['thóp hậu', 'tóp hậu', 'hậu tóp', 'đuôi tóp'],
    'Chữ nhật vát góc': ['vát góc', 'cắt góc', 'bo góc', 'xéo góc'],

    # Specific letter shapes
    'Chữ L': ['chữ l', 'hình l'],
    'Chữ L hẹp ngang': ['chữ l hẹp', 'hình l hẹp'],
    'Chữ L rộng ngang': ['chữ l rộng', 'hình l rộng'],
    'Chữ T': ['chữ t', 'hình t', 'thông ngang'],
    'Chữ U': ['chữ u', 'hình u', 'móng ngựa'],

    # Basic geometric shapes
    'Tam giác': ['tam giác'],
    'Hình quạt': ['hình quạt', 'nan quạt', 'xòe quạt'],

    # Irregular/Complex shapes
    'Đa giác từ 5 cạnh, méo mó': ['méo mó', 'méo', "móp méo"],
    'Phức tạp, nhiều góc nhọn/tù': ['nhiều góc nhọn', 'hình dáng phức tạp'],

    # Most common/default shape, checked last.
    'Chữ nhật': [
        'vuông vức', 'vuông vắn', 'vuông đẹp',
        'vuông như', 'hình chữ nhật', 'đều chằn chặn', "không lỗi phong thủy"
    ]
}

QUALITY_LEVELS = [
    # Priority 1: Structure has essentially no value (0%)
    (0.0, [
        'tặng nhà', 'bán đất tặng nhà', 'chỉ tính tiền đất',
        'đất nền', 'nhà tạm', 'chủ yếu lấy đất', 'tặng nhà'
        'có nhà nhưng không đáng giá', 'không tính giá trị nhà',
        'nhà cấp 4 cũ', 'nhà xuống cấp', 'giá trị đất là chính',
        'bán đất', 'sắp sập'
    ]),
    # Priority 2: Brand-new condition (100%)
    (1.0, [
        'mới xây', 'xây mới', 'mới xd', 'mới hoàn thiện', 'mới 100%', 'nhà mới',
        'vừa xây xong', 'mới bàn giao', 'mới nhận nhà', 'nhà rất mới',
        'chưa ở lần nào', 'nhà mới tinh', 'nhà mới toanh',
        'nhà xây mới', 'vừa hoàn thiện', 'còn thơm mùi sơn',
        'mới hoàn công', 'đảm bảo kết cấu mới',
    ]),
    # Priority 3: Old or needs significant repair (50%)
    (0.5, [
        'nhà cũ', 'cần sửa chữa', 'tiện xây mới',
        'xây lâu năm', 'xuống cấp', 'cũ nhưng ở tạm được',
        'cũ kỹ', 'nhiều năm chưa sửa', 'cần cải tạo',
        'nền móng yếu', 'cần xây lại', 
        'không có giá trị sử dụng',
    ]),
    # Priority 4: Good, well-maintained condition (85%)
    (0.85, [
        'nhà đẹp', 'còn mới', 'giữ gìn', 'full nội thất', 'thiết kế hiện đại',
        'nhà sạch sẽ', 'ở ngay', 'ở luôn', 'nhà gọn gàng', 'nội thất cao cấp',
        'không cần sửa', 'đẹp như hình', 'vào ở liền',
        'nội thất đầy đủ', 'tiện nghi', 'không lỗi phong thủy', 'không lổi phong thủy',
        'không lỗi phong thuỷ', 'không lổi phong thuỷ', 
        'còn bảo hành', 'nhà chất lượng tốt', 'xây kiên cố', 'xây chắc chắn',
    ]),
]

ALLEY_WIDTH = {
    # ngõ xe máy, ba gác 
    "ngõ xe máy": 1.5,
    "hẻm xe máy": 1.5,
    "ngách xe máy": 1.5,
    "ngõ ba gác" : 3.0,
    "hẻm ba gác" : 3.0,
    "ngõ bagac": 3.0,

    # Ngõ xe máy tránh nhau
    "xe máy tránh": 2.5,
    "ba gác tránh": 4.5, 
    "3 gác tránh": 4.5, 
    "bagac tránh": 4.5,

    # Hẻm xe hơi
    "hẻm ô tô": 5.0,
    "hẻm ôtô": 5.0,
    "hẻm ô to": 5.0,
    "hẻm ôto": 5.0,
    "hẻm o tô": 5.0,
    "hẻm otô": 5.0,
    "hẻm oto": 5.0,
    "hẻm o to": 5.0,
    "hem ô tô": 5.0,
    "hem ôtô": 5.0,
    "hem ô to": 5.0,
    "hem ôto": 5.0,
    "hem o tô": 5.0,
    "hem otô": 5.0,
    "hem oto": 5.0,
    "hem o to": 5.0,
    "hxh": 5.0,
    "hẻm xe hơi": 5.0,
    "hem xe hơi": 5.0,

    # Hẻm ô tô tải
    "hẻm xe tải": 7.5,
    "hem xe tải": 7.5,
    "hxt": 7.5,
    "ngõ xe tải": 7.5,

    # Hẻm xe tải tránh nhau
    "xe tải tránh nhau": 10,
    
    # Đỗ cửa/vào nhà
    "ô tô vào nhà": 5.0,
    "ôtô vào nhà": 5.0,
    "ô to vào nhà": 5.0,
    "ôto vào nhà": 5.0,
    "o tô vào nhà": 5.0,
    "otô vào nhà": 5.0,
    "o to vào nhà": 5.0,
    "oto vào nhà": 5.0,

    "ô tô vào tận nhà": 5.0,
    "ôtô vào tận nhà": 5.0,
    "ô to vào tận nhà": 5.0,
    "ôto vào tận nhà": 5.0,
    "o tô vào tận nhà": 5.0,
    "otô vào tận nhà": 5.0,
    "o to vào tận nhà": 5.0,
    "oto vào tận nhà": 5.0,

    "ô tô đỗ cửa": 4.0,
    "ôtô đỗ cửa": 4.0,
    "ô to đỗ cửa": 4.0,
    "ôto đỗ cửa": 4.0,
    "o tô đỗ cửa": 4.0,
    "otô đỗ cửa": 4.0,
    "o to đỗ cửa": 4.0,
    "oto đỗ cửa": 4.0,

    "ô tô đỗ ngay cửa": 4.0,
    "ôtô đỗ ngay cửa": 4.0,
    "ô to đỗ ngay cửa": 4.0,
    "ôto đỗ ngay cửa": 4.0,
    "o tô đỗ ngay cửa": 4.0,
    "otô đỗ ngay cửa": 4.0,
    "o to đỗ ngay cửa": 4.0,
    "oto đỗ ngay cửa": 4.0,

    "ô tô đỗ cổng": 4.0,
    "ôtô đỗ cổng": 4.0,
    "ô to đỗ cổng": 4.0,
    "ôto đỗ cổng": 4.0,
    "o tô đỗ cổng": 4.0,
    "otô đỗ cổng": 4.0,
    "o to đỗ cổng": 4.0,
    "oto đỗ cổng": 4.0,

    "ô tô đỗ ngay cổng": 4.0,
    "ôtô đỗ ngay cổng": 4.0,
    "ô to đỗ ngay cổng": 4.0,
    "ôto đỗ ngay cổng": 4.0,
    "o tô đỗ ngay cổng": 4.0,
    "otô đỗ ngay cổng": 4.0,
    "o to đỗ ngay cổng": 4.0,
    "oto đỗ ngay cổng": 4.0,

    'ba gác vào nhà': 3.0,
    '3 bác vào nhà': 3.0,
    'bagac vào nhà': 3.0,
    'ba gác vào tận nhà': 3.0,
    '3 bác vào tận nhà': 3.0,
    'bagac vào tận nhà': 3.0,

    'ba gác đỗ cửa': 3.0,
    '3 bác đỗ cửa': 3.0,
    'bagac đỗ cửa': 3.0,
    'ba gác đỗ ngay cửa': 3.0,
    '3 bác đỗ ngay cửa': 3.0,
    'bagac đỗ ngay cửa': 3.0,

    'ba gác đỗ cổng': 3.0,
    '3 bác đỗ cổng': 3.0,
    'bagac đỗ cổng': 3.0,
    'ba gác đỗ ngay cổng': 3.0,
    '3 bác đỗ ngay cổng': 3.0,
    'bagac đỗ ngay cổng': 3.0,

    # Ngõ ô tô tránh nhau
    "ô tô tránh": 6.0,
    "ôtô tránh": 6.0,
    "ô to tránh": 6.0,
    "ôto tránh": 6.0,
    "o tô tránh": 6.0,
    "otô tránh": 6.0,
    "o to tránh": 6.0,
    "oto tránh": 6.0,

    # Ngõ ô tô 
    "ngõ ô tô": 5.0,
    "ngõ ôtô": 5.0,
    "ngõ ô to": 5.0,
    "ngõ ôto": 5.0,
    "ngõ o tô": 5.0,
    "ngõ otô": 5.0,
    "ngõ oto": 5.0,
    "ngõ o to": 5.0,
}
