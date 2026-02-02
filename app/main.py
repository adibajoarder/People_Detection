from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
import os
import shutil

from app.utils import ensure_dirs, UPLOAD_DIR, OUTPUT_DIR, unique_filename
from app.pipeline import run_full_pipeline_single

ensure_dirs()

app = FastAPI(title="People Detection System")

# Use absolute paths so StaticFiles always points to the correct folders
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = unique_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "uploaded", "filename": filename}


@app.get("/preview/{filename}", response_class=HTMLResponse)
async def preview_page(request: Request, filename: str):
    return templates.TemplateResponse("preview.html", {"request": request, "filename": filename})


@app.get("/process/{filename}")
async def process_video(filename: str):
    input_path = os.path.join(UPLOAD_DIR, filename)

    output_filename = f"FINAL_{filename.replace('.mp4','')}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Run blocking task in threadpool
    await run_in_threadpool(run_full_pipeline_single, input_path, output_path)

    return {"status": "done", "output_video": f"/video/{output_filename}"}


@app.get("/video/{filename}")
async def stream_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        file_path,
        media_type="video/mp4",
        filename=filename,
    )


@app.get("/webcam", response_class=HTMLResponse)
async def webcam_page(request: Request):
    return templates.TemplateResponse("webcam.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
        "<rect width='64' height='64' rx='12' fill='#1f2937'/>"
        "<circle cx='32' cy='28' r='12' fill='#22d3ee'/>"
        "<rect x='18' y='40' width='28' height='8' rx='4' fill='#22d3ee'/>"
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")
