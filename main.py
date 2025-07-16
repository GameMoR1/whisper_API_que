from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core.queue import TaskQueue
from core.gpu import GPUWorker
from core.task import Task
from core.utils import log_event, get_logs, cleanup_files
from core.model_manager import ModelManager, MODEL_NAMES
from core.webhook_notifier import webhook_notifier
import torch
import os
import tempfile
import threading

app = FastAPI()
queue = TaskQueue()
logs = []
model_manager = ModelManager()

def log(msg):
    log_event(logs, msg)

# --- Предзагрузка моделей на CPU (только скачивание весов, VRAM не используется) ---
preload_thread = threading.Thread(target=model_manager.preload_all)
preload_thread.start()

# --- Запуск воркеров под каждый GPU ---
for gpu_id in range(torch.cuda.device_count()):
    worker = GPUWorker(gpu_id, queue, log)
    worker.start()

# --- Запуск webhook-нотификатора, если очередь пуста ---
webhook_url = "https://n8n.kyter.space/webhook-test/a141441b-57cf-4c44-b131-bd86cb9c3a3a"
notifier_thread = threading.Thread(target=webhook_notifier, args=(queue, webhook_url), daemon=True)
notifier_thread.start()

TEMP_DIR = os.path.abspath("temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)
cleanup_thread = threading.Thread(target=cleanup_files, args=(queue, TEMP_DIR, logs), daemon=True)
cleanup_thread.start()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    initial_prompt: str = Form(None),
    upgrade_transcribation: bool = Form(False),
    up_speed: float = Form(1.0)
):
    if model_name not in MODEL_NAMES:
        return JSONResponse({"error": "Unknown model name"}, status_code=400)

    if not model_manager.is_downloaded(model_name):
        return JSONResponse({"error": f"Модель '{model_name}' ещё не скачана. Подождите завершения загрузки."}, status_code=400)

    fd, input_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[-1], dir=TEMP_DIR)
    os.close(fd)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Получаем размер файла
    file_size = os.path.getsize(input_path)

    task = Task(
        file_path=input_path,
        filename=file.filename,
        model_name=model_name,
        initial_prompt=initial_prompt,
        up_speed=up_speed,
        upgrade_transcribation=upgrade_transcribation
    )
    task.file_size = file_size

    queue.add_task(task)
    log(f"Task {task.id} queued: {file.filename} | Model: {model_name} | File size: {file_size} bytes")

    return {"task_id": task.id}

@app.get("/api/model_status")
def api_model_status():
    return model_manager.get_status()

@app.get("/api/status/{task_id}")
def task_status(task_id: str):
    for coll in [queue.processing, queue.completed, queue.failed]:
        if task_id in coll:
            task = coll[task_id]
            return {
                "status": task.status,
                "result": task.result,
                "error": task.error,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "gpu_id": task.gpu_id,
                "filename": task.filename,
                "model_name": getattr(task, "model_name", None),
                "file_size": getattr(task, "file_size", None),
                "queue_time": getattr(task, "queue_time", None),
                "processing_time": getattr(task, "processing_time", None)
            }
    for task in queue.queue:
        if task.id == task_id:
            return {"status": task.status, "result": None}
    return JSONResponse({"error": "Task not found"}, status_code=404)

@app.get("/api/status")
def api_status():
    # Проверяем, все ли модели загружены
    all_loaded = all(model_manager.is_downloaded(name) for name in MODEL_NAMES)
    return {"status": all_loaded}

@app.get("/api/queue")
def get_queue():
    return {
        "queue": [
            {"id": t.id, "filename": t.filename, "created_at": t.created_at}
            for t in list(queue.queue)
        ],
        "processing": [
            {"id": t.id, "filename": t.filename, "gpu_id": t.gpu_id, "started_at": t.started_at}
            for t in queue.processing.values()
        ],
        "completed": [
            {"id": t.id, "filename": t.filename, "completed_at": t.completed_at}
            for t in queue.completed.values()
        ],
        "failed": [
            {"id": t.id, "filename": t.filename, "error": t.error}
            for t in queue.failed.values()
        ]
    }

@app.get("/api/gpus")
def get_gpus():
    gpus = []
    try:
        import subprocess
        smi_out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'
            ],
            encoding='utf-8'
        )
        for line in smi_out.strip().split('\n'):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 5:
                continue
            idx, name, mem_total, mem_used, util = parts
            try:
                util_val = int(util)
            except Exception:
                util_val = None
            gpus.append({
                "id": int(idx),
                "name": name,
                "memory_total_mb": int(mem_total),
                "memory_used_mb": int(mem_used),
                "utilization_gpu": util_val
            })
    except Exception:
        import torch
        for gpu_id in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(gpu_id)
            mem_total = prop.total_memory // (1024**2)
            mem_alloc = torch.cuda.memory_allocated(gpu_id) // (1024**2)
            gpus.append({
                "id": gpu_id,
                "name": prop.name,
                "memory_total_mb": mem_total,
                "memory_used_mb": mem_alloc,
                "utilization_gpu": None
            })
    return gpus

@app.get("/api/logs")
def api_logs():
    return get_logs(logs, 200)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
