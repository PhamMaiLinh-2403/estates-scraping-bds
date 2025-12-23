from pathlib import Path

# --- BROWSER CONFIGURATION ---
SELENIUM_CONFIG = {
    "headless": False,
    "uc_driver": True,
    "window_size": "1920,1080",
    "args": [
        "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
        "--disable-extensions", "--disable-infobars", "--disable-web-security",
        "--disable-background-networking", "--disable-default-apps", "--disable-sync",
        "--disable-translate", "--metrics-recording-only", "--mute-audio",
        "--no-first-run", "--no-default-browser-check", "--disable-gpu"
    ]
}

# --- SCRAPER SETTINGS ---
MAX_WORKERS = 3
BASE_URL = "https://batdongsan.com.vn"
SEARCH_PAGE_URL = f"{BASE_URL}/ban-dat"
PAGE_NUMBER = 1

SCRAPING_DETAILS_CONFIG = {
    "append_mode": True,
    "start_index": 0,
    "count": 3000,
    "stagger_mode": "random",
    "stagger_step_sec": 3.0,
    "stagger_max_sec": 3.0,
}

# --- FILE & DIRECTORY PATHS ---
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"
ADMIN_DATA_DIR = BASE_DIR / "Dữ liệu địa giới hành chính"
INFO_DATA_DIR = BASE_DIR / "Dữ liệu thông tin kỹ thuật tài sản"

URLS_OUTPUT_FILE = OUTPUT_DIR / "listing_urls.csv"
DETAILS_OUTPUT_FILE = OUTPUT_DIR / "listing_details.csv"
CLEANED_DETAILS_OUTPUT_FILE = OUTPUT_DIR / "listing_details_cleaned.xlsx"

PROVINCES_SQL_FILE = ADMIN_DATA_DIR / "provinces_20250225_2.sql"
DISTRICTS_SQL_FILE = ADMIN_DATA_DIR / "districts_20250225_2.sql"
WARDS_SQL_FILE = ADMIN_DATA_DIR / "wards_20250225_2.sql"
STREETS_SQL_FILE = ADMIN_DATA_DIR / "streets_20250225_2.sql"
INFO_FILE = INFO_DATA_DIR / "Dữ liệu thông tin kỹ thuật tài sản.xlsx"

# --- DATA EXTRACTION MAPS ---
FACADE_COUNT_MAP = [
    (r'\b3\s+mặt\s+tiền\b', 3), (r'\bba\s+mặt\s+tiền\b', 3),
    (r'\blô\s+góc\s+2\s+mặt\s+tiền\b', 2), (r'\b2\s+mặt\s+tiền\b', 2),
    (r'\bhai\s+mặt\s+tiền\b', 2), (r'\blô\s+góc\b', 2), (r'\bcăn\s+góc\b', 2),
    (r'\b1\s+mặt\s+tiền\b', 1), (r'\bmột\s+mặt\s+tiền\b', 1),
]

SHAPE_KEYWORDS = {
    'Nở hậu': ['nở hậu', 'hậu nở', 'đuôi nở', 'phía sau nở'],
    'Thóp hậu': ['thóp hậu', 'tóp hậu', 'hậu tóp', 'đuôi tóp'],
    'Chữ nhật vát góc': ['vát góc', 'cắt góc', 'bo góc', 'xéo góc'],
    'Chữ L': ['chữ l', 'hình l'],
    'Chữ L hẹp ngang': ['chữ l hẹp', 'hình l hẹp'],
    'Chữ L rộng ngang': ['chữ l rộng', 'hình l rộng'],
    'Chữ T': ['chữ t', 'hình t', 'thông ngang'],
    'Chữ U': ['chữ u', 'hình u', 'móng ngựa'],
    'Tam giác': ['tam giác'],
    'Hình quạt': ['hình quạt', 'nan quạt', 'xòe quạt'],
    'Đa giác từ 5 cạnh, méo mó': ['méo mó', 'méo', "móp méo"],
    'Phức tạp, nhiều góc nhọn/tù': ['nhiều góc nhọn', 'hình dáng phức tạp'],
    'Chữ nhật': ['vuông vức', 'vuông vắn', 'vuông đẹp', 'vuông như', 'hình chữ nhật', 'đều chằn chặn', "không lỗi phong thủy"]
}

QUALITY_LEVELS = [
    (0.0, ['tặng nhà', 'bán đất tặng nhà', 'chỉ tính tiền đất', 'đất nền', 'nhà tạm', 'chủ yếu lấy đất', 'có nhà nhưng không đáng giá', 'không tính giá trị nhà', 'nhà cấp 4 cũ', 'nhà xuống cấp', 'giá trị đất là chính', 'bán đất', 'sắp sập', "đập đi xây lại"]),
    (1.0, ['mới xây', 'xây mới', 'mới xd', 'mới hoàn thiện', 'mới 100%', 'nhà mới', 'vừa xây xong', 'mới bàn giao', 'mới nhận nhà', 'nhà rất mới', 'chưa ở lần nào', 'nhà mới tinh', 'nhà mới toanh', 'nhà xây mới', 'vừa hoàn thiện', 'còn thơm mùi sơn', 'mới hoàn công', 'đảm bảo kết cấu mới']),
    (0.5, ['nhà cũ', 'cần sửa chữa', 'tiện xây mới', 'xây lâu năm', 'xuống cấp', 'cũ nhưng ở tạm được', 'cũ kỹ', 'nhiều năm chưa sửa', 'cần cải tạo', 'nền móng yếu', 'cần xây lại', 'không có giá trị sử dụng']),
    (0.85, ['nhà đẹp', 'còn mới', 'giữ gìn', 'full nội thất', 'thiết kế hiện đại', 'nhà sạch sẽ', 'ở ngay', 'ở luôn', 'nhà gọn gàng', 'nội thất cao cấp', 'không cần sửa', 'đẹp như hình', 'vào ở liền', 'nội thất đầy đủ', 'tiện nghi', 'không lỗi phong thủy', 'không lổi phong thủy', 'không lỗi phong thuỷ', 'không lổi phong thuỷ', 'còn bảo hành', 'nhà chất lượng tốt', 'xây kiên cố', 'xây chắc chắn']),
]

ALLEY_WIDTH = {
    "ngõ xe máy": 1.5, "hẻm xe máy": 1.5, "ngách xe máy": 1.5,
    "ngõ ba gác" : 3.0, "hẻm ba gác" : 3.0, "ngõ bagac": 3.0,
    "xe máy tránh": 2.5, "ba gác tránh": 4.5, "3 gác tránh": 4.5, "bagac tránh": 4.5,
    "hẻm ô tô": 5.0, "hẻm ôtô": 5.0, "hẻm ô to": 5.0, "hẻm ôto": 5.0, "hẻm o tô": 5.0, "hẻm otô": 5.0, "hẻm oto": 5.0, "hxh": 5.0, "hẻm xe hơi": 5.0,
    "hẻm xe tải": 7.5, "hxt": 7.5, "ngõ xe tải": 7.5, "xe tải tránh nhau": 10.0,
    "ô tô vào nhà": 5.0, "ôtô vào nhà": 5.0, "ô tô đỗ cửa": 4.0, "ôtô đỗ cửa": 4.0,
    "ô tô tránh": 6.0, "ôtô tránh": 6.0, "ngõ ô tô": 5.0, "ngõ ôtô": 5.0
}

LAND_PURPOSE_DICT = {
    'đất trồng lúa|đất chuyên trồng lúa|đất lúa|đất (?:\S+\s){0,2}trồng cây hàng năm|đất (?:\S+\s){0,2}trồng cây hằng năm|(?:đất\s)?\W(?:luc|lua|lun|luk|lun|bhk|nhk|hnk|nkh)\W': 'Đất trồng cây hàng năm',
    'đất (?:trồng\s)?cây lâu năm|đất cln|cây lâu năm|(?:đất\s)?(?:lnc|lnq|lnk)': 'Đất trồng cây lâu năm',
    'đất trồng cây\W(?!hàng năm|hằng năm|lâu năm)|đất vườn': 'Đất vườn',
    'đất rừng|(?:đất\s)?\W(?:rsx|rph|rdd|rsn|rst|rsm|rsk|rsm|rpn|rpt|rpk|rpm|rdn|rdt|rdk|rdm)\W': 'Đất rừng',
    'đất nông nghiệp khác|đất nkh': 'Đất nông nghiệp khác',
    'đất ở|thổ cư|full thổ|(?:đất\s)?\W(?:ont|odt)\W': 'Đất ở',
    'đất khu công nghiệp|đất kcn|đất (?:skk|skn|skt)': 'Đất khu công nghiệp',
    'đất thương mại|đất tm\W|đất dịch vụ|đất dvu\W|đất dv\W|đất tmdv|đất kinh doanh|(?:đất\s)?\Wtmd\W': 'Đất thương mại, dịch vụ',
}