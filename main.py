from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from core.queue import TaskQueue
from core.gpu import GPUWorker
from core.task import Task
from core.audio_splitter import split_audio_channels
from core.utils import log_event, get_logs, cleanup_files
from core.model_manager import ModelManager, MODEL_NAMES
from core.webhook_notifier import webhook_notifier
from core.webhook_timer import get_webhook_timer_state
from core.config import WEBHOOK_URL
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
notifier_thread = threading.Thread(target=webhook_notifier, args=(queue, WEBHOOK_URL), daemon=True)
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
    up_speed: float = Form(1.0),
    split_roads: str = Form(None)
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

    # Получаем длительность аудио (секунды)
    import subprocess
    import json
    try:
        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', input_path
        ]
        ffprobe_out = subprocess.check_output(ffprobe_cmd, encoding='utf-8')
        duration = float(json.loads(ffprobe_out)['format']['duration'])
    except Exception:
        duration = 0.0

    # Если split_roads не задан — обычная логика
    if not split_roads:
        task = Task(
            file_path=input_path,
            filename=file.filename,
            model_name=model_name,
            initial_prompt=initial_prompt,
            up_speed=up_speed,
            upgrade_transcribation=upgrade_transcribation
        )
        task.file_size = file_size
        task.duration = duration
        queue.add_task(task)
        log(f"Task {task.id} queued: {file.filename} | Model: {model_name} | File size: {file_size} bytes")
        return {"task_id": task.id}

    # Новый режим: split_roads = "2,Клиент,Сотрудник"
    try:
        parts = split_roads.split(",")
        num_channels = int(parts[0])
        role_names = parts[1:]
        if len(role_names) != num_channels:
            return JSONResponse({"error": "split_roads: число ролей не совпадает с количеством дорожек"}, status_code=400)
    except Exception:
        return JSONResponse({"error": "split_roads: неверный формат"}, status_code=400)


    # Проверяем количество каналов
    try:
        ffprobe_channels_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels',
            '-of', 'json', input_path
        ]
        ffprobe_out = subprocess.check_output(ffprobe_channels_cmd, encoding='utf-8')
        channels = int(json.loads(ffprobe_out)['streams'][0]['channels'])
    except Exception:
        channels = 1
    if channels < num_channels:
        return JSONResponse({"error": f"В аудиофайле только {channels} канал(а/ов), а требуется {num_channels}."}, status_code=400)

    # Если не wav — конвертируем во временный wav с нужным количеством каналов
    ext = os.path.splitext(input_path)[-1].lower()
    if ext != '.wav':
        fd, wav_path = tempfile.mkstemp(suffix='.wav', dir=TEMP_DIR)
        os.close(fd)
        ffmpeg_conv_cmd = [
            'ffmpeg', '-y', '-i', input_path, '-acodec', 'pcm_s16le', '-ac', str(channels), wav_path
        ]
        try:
            subprocess.run(ffmpeg_conv_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            return JSONResponse({"error": f"Ошибка конвертации в wav: {e}"}, status_code=500)
        input_path_for_split = wav_path
    else:
        input_path_for_split = input_path

    try:
        split_paths = split_audio_channels(input_path_for_split, num_channels, role_names, TEMP_DIR)
    except Exception as e:
        return JSONResponse({"error": f"Ошибка разделения дорожек: {e}"}, status_code=500)

    # Для каждой дорожки — отдельная задача
    tasks = []
    for i, (role, path) in enumerate(zip(role_names, split_paths)):
        t = Task(
            file_path=path,
            filename=f"{role}_{file.filename}",
            model_name=model_name,
            initial_prompt=initial_prompt,
            up_speed=up_speed,
            upgrade_transcribation=upgrade_transcribation
        )
        t.file_size = os.path.getsize(path)
        t.duration = duration
        t.role = role
        tasks.append(t)
        queue.add_task(t)
        log(f"Task {t.id} queued: {file.filename} | Role: {role} | Model: {model_name}")

    # Вернуть один task_id (групповой) — для простоты, можно вернуть список
    group_id = "group_" + tasks[0].id
    # Сохраним маппинг group_id -> tasks в памяти (или можно в очередь, если нужно)
    if not hasattr(queue, "groups"):
        queue.groups = {}
    queue.groups[group_id] = [t.id for t in tasks]
    return {"task_id": group_id}

@app.get("/api/model_status")
def api_model_status():
    return model_manager.get_status()

@app.get("/api/status/{task_id}")
def task_status(task_id: str):
    # Если это group_id (split_roads)
    if hasattr(queue, "groups") and task_id in queue.groups:
        task_ids = queue.groups[task_id]
        # Собираем задачи по id из всех коллекций
        all_tasks = []
        found_ids = set()
        for coll in [queue.processing, queue.completed, queue.failed]:
            for t_id in task_ids:
                if t_id in coll and t_id not in found_ids:
                    all_tasks.append(coll[t_id])
                    found_ids.add(t_id)
        # Добавить задачи из очереди (queue.queue)
        for t in list(queue.queue):
            if t.id in task_ids and t.id not in found_ids:
                all_tasks.append(t)
                found_ids.add(t.id)
        # Если не все задачи найдены — значит, часть ещё не создана или удалена
        if len(all_tasks) < len(task_ids):
            return {"status": "pending", "result": None}
        # Если хоть одна задача не завершена — pending
        if any(t.status not in ("done", "error") for t in all_tasks):
            return {"status": "pending", "result": None}
        # Если есть ошибки — вернуть ошибку
        if any(t.status == "error" for t in all_tasks):
            errors = [t.error for t in all_tasks if t.status == "error"]
            return {"status": "error", "error": "; ".join(errors)}
        # Собрать два блока по ролям (без сортировки)
        role_results = {}
        for t in all_tasks:
            role = getattr(t, "role", None) or "role"
            segments = getattr(t, "segments", None)
            if segments:
                def format_timestamp(seconds):
                    m, s = divmod(int(seconds), 60)
                    return f"{m:02d}:{s:02d}"
                text = "\n".join(f"[{format_timestamp(seg.get('start',0))}] {seg.get('text','').strip()}" for seg in segments)
                role_results[role] = text
            else:
                role_results[role] = t.result or ""
        gpt_input = "\n\n".join(f"[{role}]\n{role_results[role]}" for role in role_results)
        # Явный промпт для диалога
        prompt = (
            "Вот расшифровка диалога между двумя ролями (например, Клиент и Сотрудник). "
            "Твоя задача — собрать из этих двух блоков настоящий диалог: "
            "раскидать реплики по времени, подписать роли, сохранить таймкоды, оформить как диалог. "
            "Если есть ошибки или неразборчивые фрагменты — пометь их.\n\n" + gpt_input
        )
        try:
            from services.transcriber import improve_transcription_with_gpt
            gpt_result = improve_transcription_with_gpt(prompt)
            return {"status": "done", "result": gpt_result}
        except Exception as e:
            return {"status": "done", "result": gpt_input + f"\n\n[Ошибка улучшения через GPT: {e}]"}

    # Обычная задача
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

@app.get("/api/webhook_timer")
def api_webhook_timer():
    return get_webhook_timer_state(queue)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
