from fastapi import FastAPI, Request, Query, Path, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from io import BytesIO
import uuid
from pydantic import BaseModel
from typing import Annotated
from fastapi.templating import Jinja2Templates
from datetime import datetime, date
import time
import pandas as pd
from tqdm import tqdm
import threading
import traceback
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from main import run_scrape_urls,run_scrape_details, run_cleaning_pipeline
from src.config import *

app = FastAPI()
templates = Jinja2Templates(directory="templates")
RESULT_STORE = {}

scrape_state = {
    "running": False,
    "last_run": None,
    "message": "Idle"
}

scrape_lock = threading.Lock()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if scrape_state["running"]:
        scrape_state["message"] = "Scraping in progress"
        return RedirectResponse("/system-busy")
    
    return templates.TemplateResponse(
        "home.html", 
        {"request": request}
    )

@app.get("/system-busy", response_class=HTMLResponse)
async def system_busy(request: Request):
    return templates.TemplateResponse(
        "system_busy.html", 
        {"request": request}
    )

@app.get("/system-status")
def system_status():
    return scrape_state

@app.post("/submit", response_class=HTMLResponse)
async def submit(
    request: Request,
    web: str = Form(...),
    start_time: datetime = Form(...),
    end_time: datetime = Form(...),
    # request_name: str = Form(...)
):
    if scrape_state["running"]:
        return RedirectResponse(
            url="/system-busy",
            status_code=303 # Forces POST to GET
        )
    start_time_display = start_time.strftime("%d/%m/%Y")
    end_time_display = end_time.strftime("%d/%m/%Y")

    df = extract_data(start_time, end_time, web)
    if df is None:
        result_text = "Không tìm thấy thông tin"
    else:
        result_text = f'Thành công thu thập thông tin. Số lượng thông tin thu thập: {df.shape[0]}'
    job_id = str(uuid.uuid4())
    RESULT_STORE[job_id] = df

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "web": web,
            "start_time": start_time_display,
            "end_time": end_time_display,
            # "request_name": request_name,
            "result_text": result_text,
            "job_id": job_id
            # "scraped_data": df
        }
    )

@app.get("/job/{job_id}")
def job_status(job_id: str):
    df = RESULT_STORE[job_id]

    if df.shape[0] == 0:
        return {"status": "failed"}

    return {
        "status": "done",
        "row_count": int(df.shape[0])
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    df = RESULT_STORE[job_id]
    if df.shape[0] == 0:
        return {"error": "File không tồn tại hoặc đã hết hạn"}
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    buffer.seek(0)

    del RESULT_STORE[job_id]

    return StreamingResponse(
        buffer, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}.xlsx"
        })

# ------- Functions ------
def extract_data(start_date, end_date, web):
    date_file = f'{OUTPUT_DIR}/{web}/{DATE_FILE}'
    print(date_file)
    df = pd.read_json(date_file, lines=True)
    df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True)
    df['end_time'] = pd.to_datetime(df['end_time'], dayfirst=True)
    try:
        start_file_index = df[df['start_time'] <= start_date].index[-1]
        print(f'Start file index: {start_file_index}')
        end_file_index = df[df['end_time'] >= end_date].index[0]
        print(f'End file index: {end_file_index}')
        print(f'{OUTPUT_DIR}/{web}/{df.iloc[start_file_index]["file_dir"]}')
        return_df = pd.read_excel(f'{OUTPUT_DIR}/{web}/{df.iloc[start_file_index]["file_dir"]}')
        print(f'Return df shape:{return_df.shape[0]}')

        for i in range(start_file_index + 1, end_file_index + 1):
            new_df = pd.read_excel(f'{OUTPUT_DIR}/{web}/{df.iloc[i]["file_dir"]}')
            return_df = pd.concat([return_df, new_df], ignore_index=True)
        
        print("We're here")
        
        if web == 'Batdongsan':
            return_df['Thời điểm giao dịch/rao bán'] = pd.to_datetime(return_df['Thời điểm giao dịch/rao bán'], dayfirst=True)
            return_df = return_df[(return_df['Thời điểm giao dịch/rao bán'] >= start_date) & (return_df['Thời điểm giao dịch/rao bán'] <= end_date)]
            return_df['Thời điểm giao dịch/rao bán'] = return_df['Thời điểm giao dịch/rao bán'].dt.strftime("%d/%m/%Y")
            return_df.drop_duplicates(subset=['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố', 'Giá rao bán/giao dịch', 'Giá ước tính', 'Đơn giá đất', 'Lợi thế kinh doanh', 'Số tầng công trình', 'Tổng diện tích sàn', 'Đơn giá xây dựng', 'Chất lượng còn lại', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất'], inplace=True)
        else:
            print("We're here")
            print(return_df.shape[0])
            print(return_df.columns)
            return_df.drop_duplicates(subset=['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố', 'Giá rao bán/giao dịch', 'Giá ước tính', 'Đơn giá đất', 'Lợi thế kinh doanh', 'Số tầng công trình', 'Tổng diện tích sàn', 'Chất lượng còn lại', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất'], inplace=True)
        # return_df.drop_duplicates(subset=['Tỉnh/Thành phố', 'Thành phố/Quận/Huyện/Thị xã', 'Xã/Phường/Thị trấn', 'Đường phố', 'Giá rao bán/giao dịch', 'Giá ước tính', 'Đơn giá đất', 'Lợi thế kinh doanh', 'Số tầng công trình', 'Tổng diện tích sàn', 'Đơn giá xây dựng', 'Chất lượng còn lại', 'Diện tích đất (m2)', 'Kích thước mặt tiền (m)', 'Kích thước chiều dài (m)', 'Số mặt tiền tiếp giáp', 'Hình dạng', 'Độ rộng ngõ/ngách nhỏ nhất (m)', 'Khoảng cách tới trục đường chính (m)', 'Mục đích sử dụng đất'], inplace=True)
        
    except:
        print("File not found")
        return_df = pd.DataFrame()
    return return_df

# def test_auto():
#     scrape_state["running"] = True
#     print("Start testing")

# def end_test_auto():
#     scrape_state["running"] = False
#     print('End testing')

def weekly_pipeline():
    with scrape_lock:
        if scrape_state["running"]:
            return

        scrape_state["running"] = True
        scrape_state["message"] = "Scraping started"

    try:
        # Step 1: scrape links
        scrape_state["message"] = "Scraping links..."
        run_scrape_urls()

        # Step 2: scrape details from link
        scrape_state["message"] = "Scraping details..."
        run_scrape_details()

        # Step 3: clean data
        scrape_state["message"] = "Cleaning data..."
        run_cleaning_pipeline(mode="house")

        scrape_state["last_run"] = datetime.now().isoformat()
        scrape_state["message"] = "Pipeline completed successfully"

    except Exception as e:
        scrape_state["message"] = "Pipeline failed"
        scrape_state["error"] = str(e)
        traceback.print_exc()

    finally:
        scrape_state["running"] = False

scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
scheduler.add_job(
    weekly_pipeline,
    CronTrigger(
        day_of_week='fri',
        hour=21,
        minute=0
    ),
    id="weekly_scrape",
    replace_existing=True
)

scheduler.start()


# scheduler = BackgroundScheduler(timezone="Asia/Ho_Chi_Minh")
# scheduler.add_job(
#     end_test_auto,
#     CronTrigger(
#         day_of_week='tue',
#         hour=11,
#         minute=0
#     ),
#     id="weekly_scrape",
#     replace_existing=True
# )

# scheduler.start()