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
MAX_WORKERS = 2

# Scraping details configuration
SCRAPING_DETAILS_CONFIG = {
    "append_mode": True,
    "start_index": 3001,
    "count": 1000,
    "stagger_mode": "random",
    "stagger_step_sec": 3.0,
    "stagger_max_sec": 3.0,
}

# Scraping configuration
BASE_URL = "https://batdongsan.com.vn"
SEARCH_PAGE_URL = f"{BASE_URL}/ban-nha-rieng"
PAGE_NUMBER = 1  # Starting page number for scraping

# Helper URLs
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# File path settings
OUTPUT_DIR = "output"
URLS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_urls.csv"
DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details.csv"
CLEANED_DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details_cleaned.csv"
FINAL_OUTPUT = f"{OUTPUT_DIR}/final_output.xlsx"

# Administrative data paths
ADMIN_DATA_DIR = "Dữ liệu địa giới hành chính"
PROVINCES_SQL_FILE = f"{ADMIN_DATA_DIR}/provinces_20250225_2.sql"
DISTRICTS_SQL_FILE = f"{ADMIN_DATA_DIR}/districts_20250225_2.sql"
WARDS_SQL_FILE = f"{ADMIN_DATA_DIR}/wards_20250225_2.sql"
STREETS_SQL_FILE = f"{ADMIN_DATA_DIR}/streets_20250225_2.sql"


# -- Word maps for cleaning -- 
FACADE_COUNT_MAP = [
        # Keywords for 3 facades
        (r'\b3\s+mặt\s+tiền\b', 3),
        (r'\bba\s+mặt\s+tiền\b', 3),
        (r'\blô\s+góc\s+3\s+mặt\b', 3),
        (r'\b3\s+mặt\s+thoáng\b', 3),

        # Keywords for 2 facades (most common variant after 1)
        (r'\blô\s+góc\s+2\s+mặt\s+tiền\b', 2),
        (r'\b2\s+mặt\s+tiền\b', 2),
        (r'\bhai\s+mặt\s+tiền\b', 2),
        (r'\blô\s+góc\b', 2),  # "lô góc" strongly implies 2 facades
        (r'\bcăn\s+góc\b', 2),
        (r'\b2\s+mặt\s+thoáng\b', 2),

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
        'vuông như', 'hình chữ nhật', 'đều chằn chặn'
    ]
}

QUALITY_LEVELS = [
    # Priority 1: Structure has essentially no value (0%)
    (0.0, [
        'tặng nhà', 'bán đất tặng nhà', 'chỉ tính tiền đất',
        'đất nền', 'nhà tạm', 'chủ yếu lấy đất', 'tặng nhà'
        'có nhà nhưng không đáng giá', 'không tính giá trị nhà',
        'nhà cấp 4 cũ', 'nhà xuống cấp', 'giá trị đất là chính',
        'bán đất'
    ]),
    # Priority 2: Old or needs significant repair (50%)
    (0.5, [
        'nhà cũ', 'nhà nát', 'cần sửa chữa', 'tiện xây mới',
        'xây lâu năm', 'xuống cấp', 'cũ nhưng ở tạm được',
        'cũ kỹ', 'nhiều năm chưa sửa', 'cần cải tạo',
        'nền móng yếu', 'sắp sập', 'cần xây lại', 
        'không có giá trị sử dụng',
    ]),
    # Priority 3: Good, well-maintained condition (85%)
    (0.85, [
        'nhà đẹp', 'còn mới', 'giữ gìn', 'full nội thất', 'thiết kế hiện đại',
        'nhà sạch sẽ', 'ở ngay', 'nhà gọn gàng', 'nội thất cao cấp',
        'không cần sửa', 'đẹp như hình', 'vào ở liền',
        'nội thất đầy đủ', 'tiện nghi', 'nhà không lỗi phong thủy',
        'còn bảo hành', 'nhà chất lượng tốt',
    ]),
    # Priority 4: Brand-new condition (100%)
    (1.0, [
        'mới xây', 'mới hoàn thiện', 'mới 100%', 'nhà mới keng',
        'vừa xây xong', 'mới bàn giao', 'mới nhận nhà', 'nhà rất mới',
        'chưa ở lần nào', 'nhà mới tinh', 'nhà mới toanh',
        'nhà xây mới', 'vừa hoàn thiện', 'còn thơm mùi sơn',
        'mới hoàn công', 'nhà xây kiên cố', 'đảm bảo kết cấu mới',
    ]),   
]

ALLEY_WIDTH = {
    # ngõ xe máy, ba gác 
    "ngõ xe máy": 1.5,
    "hẻm xe máy": 1.5,
    "ngách xe máy": 1.5,
    "ngõ ba gác" : 1.5,
    "hẻm ba gác" : 1.5,

    # Ngõ xe máy tránh nhau
    "xe máy tránh": 2.5,
    "ba gác tránh": 2.5, 
    "3 gác tránh": 2.5, 

    # Ngõ ô tô 
    "ngõ ô tô": 3.0,
    "hẻm ô tô": 3.0,


    # Ngõ ô tô tránh nhau
    "ô tô tránh": 5.0,

    # Hẻm ô tô tải
    "hẻm xe tải": 7.5,
    "ngõ xe tải": 7.5,

    # Hẻm xe tải tránh nhau
    "xe tải tránh nhau": 10
}
