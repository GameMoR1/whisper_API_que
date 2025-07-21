from core.config import WEBHOOK_INTERVAL, WEBHOOK_ENABLED

WEBHOOK_TIMER_STATE = {"last_empty_time": None}

def get_webhook_timer_state(queue):
    import time
    now = time.time()
    with queue.lock:
        is_empty = not queue.queue and not queue.processing
    if is_empty:
        if WEBHOOK_TIMER_STATE["last_empty_time"] is None:
            WEBHOOK_TIMER_STATE["last_empty_time"] = now
        elapsed = now - WEBHOOK_TIMER_STATE["last_empty_time"]
        remaining = max(0, WEBHOOK_INTERVAL - elapsed)
    else:
        WEBHOOK_TIMER_STATE["last_empty_time"] = None
        remaining = WEBHOOK_INTERVAL
    return {
        "enabled": WEBHOOK_ENABLED,
        "remaining": int(remaining)
    }
