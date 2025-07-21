import threading
import time
import requests
from core.config import WEBHOOK_ENABLED, WEBHOOK_INTERVAL

def webhook_notifier(queue, url, interval=WEBHOOK_INTERVAL):  # 600 секунд = 10 минут
    while True:
        time.sleep(interval)
        if not WEBHOOK_ENABLED:
            continue
        with queue.lock:
            is_empty = not queue.queue and not queue.processing
        if is_empty:
            try:
                requests.get(url, timeout=10)
            except Exception:
                pass
