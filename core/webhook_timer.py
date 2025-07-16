from core.webhook_notifier import WEBHOOK_ENABLED
import time

# Время ожидания до вызова webhook (секунд)
WEBHOOK_INTERVAL = 600

_last_empty_time = None

def get_webhook_timer_state(queue):
    global _last_empty_time
    now = time.time()
    with queue.lock:
        is_empty = not queue.queue and not queue.processing
    if is_empty:
        if _last_empty_time is None:
            _last_empty_time = now
        elapsed = now - _last_empty_time
        remaining = max(0, WEBHOOK_INTERVAL - elapsed)
    else:
        _last_empty_time = None
        remaining = WEBHOOK_INTERVAL
    return {
        "enabled": WEBHOOK_ENABLED,
        "remaining": int(remaining)
    }
