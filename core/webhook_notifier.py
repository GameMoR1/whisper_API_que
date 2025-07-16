import threading
import time
import requests

WEBHOOK_ENABLED = True

def webhook_notifier(queue, url, interval=60):
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
