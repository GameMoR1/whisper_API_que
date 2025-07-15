import os
import time
from datetime import datetime, timedelta

def log_event(logs, msg):
    logs.append({"time": datetime.utcnow().isoformat(), "msg": msg})
    if len(logs) > 5000:
        logs.pop(0)

def get_logs(logs, n=200):
    return logs[-n:]

def cleanup_files(queue, temp_dir, logs):
    while True:
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=48)
        # Очистка завершённых и ошибочных задач старше 48 часов
        for coll in [queue.completed, queue.failed]:
            to_remove = []
            for task_id, task in list(coll.items()):
                if task.completed_at and task.completed_at < cutoff:
                    if os.path.exists(task.file_path):
                        try:
                            os.remove(task.file_path)
                        except Exception:
                            pass
                    to_remove.append(task_id)
            for task_id in to_remove:
                del coll[task_id]
                log_event(logs, f"Task {task_id} deleted as old")
        # Очистка временных файлов, которых нет в активных задачах
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            try:
                st = os.stat(fpath)
                mtime = datetime.utcfromtimestamp(st.st_mtime)
                if mtime < cutoff:
                    os.remove(fpath)
            except Exception:
                pass
        time.sleep(3600)
