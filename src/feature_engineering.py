import pandas as pd
from typing import Dict, Any, Optional
from .cleaning import DataCleaner


class FeatureEngineer:
    """
    Static collection of feature engineering helpers.
    This class takes the cleaned data and adds business-specific features.
    """

    @staticmethod
    def get_location_category(row: Dict[str, Any]) -> Optional[str]:
        """
        Determines the asset's position category (VT1-VT4) based on distance
        to the main road and alley width.
        """
        distance = row.get('Khoảng cách tới trục đường chính (m)')
        alley_width = row.get('Độ rộng ngõ/ngách nhỏ nhất (m)')

        # Cannot determine VT without distance to the main road.
        if pd.isna(distance):
            return None

        if distance == 0:
            return "VT1"

        # If distance > 0, alley width is required to classify further.
        if pd.isna(alley_width):
            return None

        if alley_width >= 3.5:
            return "VT2"
        elif 2 <= alley_width < 3.5:
            return "VT3"
        elif alley_width < 2:
            return "VT4"

        return None

    @staticmethod
    def calculate_business_advantage(row: Dict[str, Any]) -> str:
        """
        Calculates the 'Lợi thế kinh doanh' (Business Advantage) based on
        the asset's location category (VT) and its district type (Quận/Huyện).
        """
        location_cate = FeatureEngineer.get_location_category(row)
        district_name = row.get('Thành phố/Quận/Huyện/Thị xã')

        if not location_cate or not isinstance(district_name, str):
            return "Kém"

        is_quan = "quận" in district_name.lower()
        is_huyen = not is_quan

        if location_cate in ["VT1", "VT2"] and is_quan:
            return "Tốt"
        if location_cate == "VT1" and is_huyen:
            return "Khá"
        if (location_cate == "VT2" and is_huyen) or (location_cate == "VT3"):
            return "Trung bình"

        return "Kém"

    @staticmethod
    def calculate_estimated_price(row):
        """
        Calculate the estimated price of the property.
        """
        total_price = row.get('Giá rao bán/giao dịch')

        if not total_price:
            return None

        return round(total_price * 0.98, 2)

    @staticmethod
    def calculate_land_unit_price(row: Dict[str, Any]) -> Optional[float]:
        """
        Calculates the land unit price per square meter.

        Formula:
        Đơn giá đất = (Giá rao bán - Đơn giá nhà) / Diện tích đất
        where, Đơn giá nhà = Đơn giá xây dựng * Tổng diện tích sàn * Chất lượng còn lại
        and, Tổng diện tích sàn is pre-calculated in the cleaning step.
        """
        total_price = row.get('Giá rao bán/giao dịch')
        construction_cost_per_sqm = row.get('Đơn giá xây dựng')
        land_area = row.get('Diện tích đất (m2)')
        remaining_quality = row.get('Chất lượng còn lại')
        total_floor_area = row.get('Tổng diện tích sàn')

        # Check for missing essential values, including the pre-calculated total_floor_area
        if pd.isna(total_price) or pd.isna(construction_cost_per_sqm) or \
                pd.isna(land_area) or pd.isna(remaining_quality) or \
                pd.isna(total_floor_area):
            return None

        # Handle edge cases to prevent errors
        if land_area <= 0 or total_floor_area <= 0:
            return None

        # The approximation logic is now handled in `extract_built_area`,
        # so we directly use the `Tổng diện tích sàn` value.

        # Calculate total building value (Đơn giá nhà)
        building_value = construction_cost_per_sqm * total_floor_area * remaining_quality

        # Land value cannot be negative or zero in this context
        if building_value >= total_price:
            return None

        land_value = total_price - building_value
        land_unit_price = land_value / land_area

        return round(land_unit_price, 2)
    
    @staticmethod
    def fill_built_area(row: Dict[str, Any]) -> Optional[float]:
        if pd.isna(row['Tổng diện tích sàn']):
            if pd.isna(row['Diện tích đất (m2)']) == False and pd.isna(row['Số tầng công trình']) == False:
                return row['Diện tích đất (m2)'] * row['Số tầng công trình']
            return None
        else:
            return row['Tổng diện tích sàn']