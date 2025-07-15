import threading
from collections import deque

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
            self.completed[task.id] = task
            self.processing.pop(task.id, None)

    def mark_failed(self, task, error):
        with self.lock:
            task.status = 'error'
            task.error = error
            from datetime import datetime
            task.completed_at = datetime.utcnow()
            self.failed[task.id] = task
            self.processing.pop(task.id, None)
