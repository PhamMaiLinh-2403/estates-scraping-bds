from fastapi import FastAPI, Request, Query, Path, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Annotated
from fastapi.templating import Jinja2Templates
from datetime import date

app = FastAPI()

templates = Jinja2Templates(directory="templates")

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
    start_time: date = Form(...),
    end_time: date = Form(...),
    request_name: str = Form(...)
):
    start_time = start_time.strftime("%d/%m/%Y")
    end_time = end_time.strftime("%d/%m/%Y")
    print(web, start_time, end_time, request_name)
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "web": web,
            "start_time": start_time,
            "end_time": end_time,
            "request_name": request_name
        }
    )

# ------- Functions ------
