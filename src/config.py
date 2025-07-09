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
MAX_WORKERS = 2  # Number of parallel threads for scraping details

SCRAPING_DETAILS_CONFIG = {
    "append_mode": False,
    "start_index": 20000,
    "count": 5000,
    "stagger_mode": "random",
    "stagger_step_sec": 2.0,
    "stagger_max_sec": 3.0,
}

# Target-specific URLs
BASE_URL = "https://batdongsan.com.vn"
SEARCH_PAGE_URL = f"{BASE_URL}/ban-nha-rieng"

# File path settings
OUTPUT_DIR = "output"
URLS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_urls.csv"
DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details.csv"
CLEANED_DETAILS_OUTPUT_FILE = f"{OUTPUT_DIR}/listing_details_cleaned.csv"
FEATURE_ENGINEERED_OUTPUT_FILE = f"{OUTPUT_DIR}/feature_engineered_listings.csv"

# Final columns of output
FINAL_COLUMNS = [
    'Tỉnh/Thành phố',
    'Thành phố/Quận/Huyện/Thị xã',
    'Xã/Phường/Thị trấn',
    'Đường phố',
    'Chi tiết',
    'Nguồn thông tin',
    'Tình trạng giao dịch',
    'Thời điểm giao dịch/rao bán',
    'Thông tin liên hệ',
    'Giá rao bán/giao dịch',
    'Giá ước tính',
    'Loại đơn giá (đ/m2 hoặc đ/m ngang)',
    'Đơn giá đất',
    'Lợi thế kinh doanh',
    'Số tầng công trình',
    'Đơn giá xây dựng',
    'Năm xây dựng',
    'Chất lượng còn lại',
    'Diện tích đất (m2)',
    'Kích thước mặt tiền (m)',
    'Kích thước chiều dài (m)',
    'Số mặt tiền tiếp giáp',
    'Hình dạng',
    'Độ rộng ngõ/ngách nhỏ nhất (m)',
    'Khoảng cách tới trục đường chính (m)',
    'Mục đích sử dụng đất',
    'Yếu tố khác',
    'Tọa độ (vĩ độ)',
    'Tọa độ (kinh độ)',
    'Hình ảnh của bài đăng'
]


# --- JSON dicts for cleaning task ---

# Land shape dict: from the most specific shape to the most general shape
SHAPE_KEYWORDS = {
    # Specific, highly desirable/undesirable features are checked first.
    'Nở hậu': ['nở hậu'],
    'Thóp hậu': ['thóp hậu', 'tóp hậu'],
    'Chữ nhật vát góc': ['vát góc', 'cắt góc'],

    # Specific letter shapes
    'Chữ L': ['chữ l', 'hình l'],
    'Chữ L hẹp ngang': ['chữ l hẹp', 'hình l hẹp'],
    'Chữ L rộng ngang': ['chữ l rộng', 'hình l rộng'],
    'Chữ T': ['chữ t', 'hình t'],
    'Chữ U': ['chữ u', 'hình u'],

    # Basic geometric shapes
    'Tam giác': ['tam giác'],
    'Hình quạt': ['hình quạt', 'nan quạt'],

    # Irregular/Complex shapes
    'Đa giác từ 5 cạnh, méo mó': ['méo mó', 'méo'],
    'Phức tạp, nhiều góc nhọn/tù': ['phức tạp'],

    # Most common/default shape, checked last.
    'Chữ nhật': [
        'vuông vức', 'vuông vắn', 'vuông đẹp',
        'vuông như', 'hình chữ nhật', 'đều chằn chặn'
    ]
}

# List for facade count mapping
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
        (r'\bcăn\s+góc\b', 2), # Synonym for corner lot
        (r'\b2\s+mặt\s+thoáng\b', 2),

        # Keywords explicitly stating 1 facade (checked last)
        (r'\b1\s+mặt\s+tiền\b', 1),
        (r'\bmột\s+mặt\s+tiền\b', 1),
    ]

# List for extracting features
FEATURE_KEYWORDS = [
    # On-Property Features
    'sân vườn', 'bể bơi', 'hồ bơi', 'gara', 'ga ra', 'hầm để xe', 'chỗ đỗ ô tô',
    'thang máy', 'giếng trời', 'sân thượng', 'full nội thất',

    # Proximity & Location
    'gần chợ', 'gần siêu thị', 'gần trường', 'gần bệnh viện', 'gần công viên',
    'gần hồ', 'gần trung tâm', 'cách chợ', 'cách trường', 'cách mặt phố',
    'cách đường',

    # Qualitative / Atmospheric Features
    'view đẹp', 'view hồ', 'view sông', 'view thoáng', 'ba mặt thoáng',
    'hai mặt thoáng', 'thoáng mát', 'thoáng đãng', 'sáng sủa', 'yên tĩnh',
    'an ninh tốt',

    # Use & Potential
    'kinh doanh', 'cho thuê', 'dòng tiền', 'văn phòng', 'kd', 'ô tô tránh',
    'ô tô vào', 'phân lô',
]

# Construction cost dict
CONSTRUCTION_COST_MAP = {
    'nhà_cấp_4': 4000000,
    'nhà_1_tầng_btct': 6275876,
    'nhà_gte_2_tầng_có_hầm': 9504604,
    'nhà_gte_2_tầng_không_hầm': 8221171,
    'biệt_thự': 10510920,
    'biệt_thự_có_hầm': 12848184,
}

# Quality map
QUALITY_LEVELS = [
    # Priority 1: Structure has essentially no value (0%)
    (0.0, [
        'tặng nhà', 'bán đất tặng nhà', 'chỉ tính tiền đất',
        'đất nền', 'nhà tạm', 'chủ yếu lấy đất', 'tặng nhà'
        'có nhà nhưng không đáng giá', 'không tính giá trị nhà',
        'nhà cấp 4 cũ', 'nhà xuống cấp', 'giá trị đất là chính',
    ]),

    # Priority 2: Brand-new condition (100%)
    (1.0, [
        'mới xây', 'mới hoàn thiện', 'mới 100%', 'nhà mới keng',
        'vừa xây xong', 'mới bàn giao', 'mới nhận nhà',
        'chưa ở lần nào', 'nhà mới tinh', 'nhà mới toanh',
        'nhà xây mới', 'vừa hoàn thiện', 'còn thơm mùi sơn',
        'hoàn công', 'nhà xây kiên cố', 'đảm bảo kết cấu mới',
    ]),

    # Priority 3: Old or needs significant repair (50%)
    (0.5, [
        'nhà cũ', 'nhà nát', 'cần sửa chữa', 'tiện xây mới',
        'xây lâu năm', 'xuống cấp', 'cũ nhưng ở tạm được',
        'cũ kỹ', 'nhiều năm chưa sửa', 'cần cải tạo',
        'nền móng yếu', 'sắp sập', 'cần xây lại',
        'nhà cấp 4 cũ kỹ', 'không có giá trị sử dụng',
    ]),

    # Priority 4: Good, well-maintained condition (85%)
    (0.85, [
        'nhà đẹp', 'còn mới', 'giữ gìn', 'full nội thất', 'thiết kế hiện đại',
        'nhà sạch sẽ', 'ở ngay', 'nhà gọn gàng', 'nội thất cao cấp',
        'không cần sửa', 'nhà rất mới', 'đẹp như hình', 'vào ở liền',
        'nội thất đầy đủ', 'tiện nghi', 'nhà không lỗi phong thủy',
        'còn bảo hành', 'nhà chất lượng tốt',
    ]),
]

DEFAULT_QUALITY = 0.75

STREET_PREFIXES = ("đường ", "phố ", "đại lộ ", "quốc lộ ")
DETAIL_PREFIXES = ("số ", "ngõ ", "hẻm ", "kiệt ", "ngách ", "sn ", "hxh ", "no. ")