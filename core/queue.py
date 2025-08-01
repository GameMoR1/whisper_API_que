import threading
from collections import deque
from core.logger import send_log_data
from core.utils import log_event

class TaskQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = deque()
        self.processing = {}
        self.completed = {}
        self.failed = {}

    def add_task(self, task):
        with self.lock:
            self.queue.append(task)

    def get_next_task(self):
        with self.lock:
            if self.queue:
                return self.queue.popleft()
            return None

    def mark_processing(self, task, gpu_id):
        with self.lock:
            task.status = 'processing'
            task.gpu_id = gpu_id
            from datetime import datetime
            task.started_at = datetime.utcnow()
            self.processing[task.id] = task

    def mark_completed(self, task, result):
        with self.lock:
            task.status = 'done'
            task.result = result
            from datetime import datetime
            task.completed_at = datetime.utcnow()
            # Вычисляем queue_time и processing_time
            if task.started_at and task.created_at:
                task.queue_time = (task.started_at - task.created_at).total_seconds()
            if task.completed_at and task.started_at:
                task.processing_time = (task.completed_at - task.started_at).total_seconds()
            self.completed[task.id] = task
            self.processing.pop(task.id, None)
        # Логирование после выхода из lock
        try:
            log_data = {
                "filename": task.filename,
                "duration": float(getattr(task, "duration", 0.0)),
                "size": int(task.file_size or 0),
                "received_at": task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else None,
                "queue_time": float(task.queue_time or 0.0),
                "process_time": float(task.processing_time or 0.0),
                "text": str(task.result or "")
            }
            from main import logs as global_logs
            send_log_data(log_data, logs=global_logs)
        except Exception as e:
            from main import logs as global_logs
            log_event(global_logs, f"Ошибка логирования: {e}")

    def mark_failed(self, task, error):
        with self.lock:
            task.status = 'error'
            task.error = error
            from datetime import datetime
            task.completed_at = datetime.utcnow()
            self.failed[task.id] = task
            self.processing.pop(task.id, None)
