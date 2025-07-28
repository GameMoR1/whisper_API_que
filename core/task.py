import uuid
from datetime import datetime

class TaskStatus:
    QUEUED = 'queued'
    PROCESSING = 'processing'
    DONE = 'done'
    ERROR = 'error'

class Task:
    def __init__(self, file_path, filename, model_name, initial_prompt, up_speed, upgrade_transcribation):
        self.id = str(uuid.uuid4())
        self.file_path = file_path
        self.filename = filename
        self.model_name = model_name
        self.initial_prompt = initial_prompt
        self.up_speed = up_speed
        self.upgrade_transcribation = upgrade_transcribation
        self.status = TaskStatus.QUEUED
        self.result = None
        self.error = None
        self.gpu_id = None
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.file_size = None  # размер файла в байтах
        self.queue_time = None  # время ожидания в очереди в секундах
        self.processing_time = None  # время выполнения задачи в секундах
        self.role = None  # имя роли (если split_roads)
        self.segments = None  # исходные сегменты whisper (для склейки по таймкодам)
