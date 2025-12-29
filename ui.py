from fastapi import FastAPI, Request, Query, Path, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from io import StringIO
from pydantic import BaseModel
from typing import Annotated
from fastapi.templating import Jinja2Templates
from datetime import datetime, date
import pandas as pd
from tqdm import tqdm
app = FastAPI()
templates = Jinja2Templates(directory="templates")
DATE_FILE = 'output/date.jsonl' # sẽ có các cột start_time, end_time, file_dir

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html", 
        {"request": request}
    )

@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    web: str = Form(...),
    start_time: datetime = Form(...),
    end_time: datetime = Form(...),
    request_name: str = Form(...)
):
    start_time_display = start_time.strftime("%d/%m/%Y")
    end_time_display = end_time.strftime("%d/%m/%Y")
    df = extract_data(start_time, end_time)
    if df.shape[0] == 0:
        result_text = "Không tìm thấy thông tin"
    else:
        result_text = f'Thành công thu thập thông tin. Số lượng thông tin thu thập: {df.shape[0]}'
    print(result_text)
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "web": web,
            "start_time": start_time_display,
            "end_time": end_time_display,
            "request_name": request_name,
            "result_text": result_text
            # "scraped_data": df
        }
    )

# ------- Functions ------
def extract_data(start_date, end_date):
    df = pd.read_json(DATE_FILE, lines=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    print(df)
    try:
        start_file_index = df[df['start_time'] <= start_date].index[-1]
        end_file_index = df[df['end_time'] >= end_date].index[0]

        print('I can extract here')

        print(start_file_index, end_file_index)
        print(df.iloc[start_file_index]['file_dir'])
        return_df = pd.read_excel(df.iloc[start_file_index]['file_dir'])
        print(f'Current shape of the df: {return_df.shape[0]}')
        for i in range(start_file_index + 1, end_file_index + 1):
            new_df = pd.read_excel(df.iloc[i]['file_dir'], encoding='utf-8')
            return_df = pd.concat([return_df, new_df], ignore_index=True)
         
        return_df['Thời điểm giao dịch/rao bán'] = pd.to_datetime(return_df['Thời điểm giao dịch/rao bán'], dayfirst=True)
        return_df = return_df[(return_df['Thời điểm giao dịch/rao bán'] >= start_date) & (return_df['Thời điểm giao dịch/rao bán'] <= end_date)]
        print(f'Shape after duplicates: {return_df.shape[0]}')
        return_df.drop_duplicates(subset=['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố', 'Giá rao bán/giao dịch', 'Giá ước tính', 'Đơn giá đất', 'Lợi thế kinh doanh', 'Số tầng công trình', 'Tổng diện tích sàn', 'Đơn giá xây dựng', 'Chất lượng còn lại', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất'], inplace=True)
        
    except:
        return_df = pd.DataFrame()
    return return_df