import threading
import torch
import time
from services.transcriber import transcribe_audio

class GPUWorker(threading.Thread):
    def __init__(self, gpu_id, queue, log_func):
        super().__init__(daemon=True)
        self.gpu_id = gpu_id
        self.queue = queue
        self.log = log_func

    def run(self):
        while True:
            try:
                task = None
                with self.queue.lock:
                    if self.queue.queue:
                        task = self.queue.queue.popleft()
                if task:
                    # Время начала обработки
                    from datetime import datetime
                    task.started_at = datetime.utcnow()
                    # Время ожидания в очереди
                    task.queue_time = (task.started_at - task.created_at).total_seconds()
                    self.queue.mark_processing(task, self.gpu_id)
                    self.log(f"Task {task.id} started on GPU {self.gpu_id} | Model: {task.model_name} | Queue time: {task.queue_time:.2f}s | File size: {task.file_size} bytes")
                    try:
                        # Установка девайса перед запуском задачи; ошибки не должны убивать поток
                        torch.cuda.set_device(self.gpu_id)
                        result, processing_time = transcribe_audio(task, self.gpu_id)
                        task.processing_time = processing_time
                        self.queue.mark_completed(task, result)
                        self.log(f"Task {task.id} completed on GPU {self.gpu_id} | Model: {task.model_name} | Processing time: {processing_time:.2f}s | Queue time: {task.queue_time:.2f}s | File size: {task.file_size} bytes")
                    except Exception as e:
                        self.queue.mark_failed(task, str(e))
                        self.log(f"Task {task.id} failed on GPU {self.gpu_id}: {e} | Model: {task.model_name} | Queue time: {task.queue_time:.2f}s | File size: {task.file_size} bytes")
                else:
                    time.sleep(1)
            except Exception as loop_error:
                # Любая непредвиденная ошибка в цикле не должна останавливать воркер
                try:
                    self.log(f"GPU worker {self.gpu_id} loop error: {loop_error}")
                except Exception:
                    pass
                time.sleep(1)
