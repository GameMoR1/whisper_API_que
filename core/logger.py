import requests
from core.config import LOGGER_API_URL

def send_log_data(log_data: dict):
    if not LOGGER_API_URL:
        return
    try:
        requests.post(LOGGER_API_URL, json=log_data, timeout=10)
    except Exception:
        pass
