from core.config import MODEL_NAMES
import whisper
import threading


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.status = {name: "not_downloaded" for name in MODEL_NAMES}

    def preload_all(self):
        threads = []
        for name in MODEL_NAMES:
            t = threading.Thread(target=self._download_model, args=(name,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def _download_model(self, model_name):
        with self.lock:
            self.status[model_name] = "downloading"
        try:
            # Скачиваем веса на диск (на CPU)
            whisper.load_model(model_name, device="cpu")
            with self.lock:
                self.status[model_name] = "downloaded"
        except Exception:
            with self.lock:
                self.status[model_name] = "error"

    def is_downloaded(self, model_name):
        with self.lock:
            return self.status.get(model_name) == "downloaded"

    def get_status(self):
        with self.lock:
            return self.status.copy()
