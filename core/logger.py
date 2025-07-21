import requests
from core.config import LOGGER_API_URL
from core.utils import log_event

def send_log_data(log_data: dict, logs=None):
    if not LOGGER_API_URL:
        if logs is not None:
            log_event(logs, "LOGGER_API_URL не задан, логирование не выполнено.")
        return
    try:
        resp = requests.post(LOGGER_API_URL, json=log_data, timeout=10)
        if resp.status_code == 200:
            if logs is not None:
                log_event(logs, f"Лог отправлен на API: {LOGGER_API_URL}")
        else:
            if logs is not None:
                log_event(logs, f"Ошибка логирования ({resp.status_code}): {resp.text}")
    except Exception as e:
        if logs is not None:
            log_event(logs, f"Ошибка логирования: {e}")
