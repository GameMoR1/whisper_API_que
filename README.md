
# 🎤 Whisper Transcription API

<p align="center">
  <a href="./API_REFERENCE.md" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/API%20Reference-%F0%9F%93%96-blue?style=for-the-badge" alt="API Reference"/>
  </a>
</p>

**Много-GPU асинхронный API для транскрибации аудио с поддержкой очереди, автозагрузки моделей и улучшения текста через GPT.**

---


## 🚀 Быстрый старт

### 🪟 Windows

1. **Установите Python 3.8+**
   - [Скачать Python](https://www.python.org/downloads/windows/)

2. **Склонируйте репозиторий и перейдите в папку проекта:**
   ```powershell
   git clone https://github.com/GameMoR1/whisper_API_que.git
   cd whisper_API_que
   ```

3. **Создайте и активируйте виртуальное окружение:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. **Установите зависимости:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Запустите сервер:**
   ```powershell
   uvicorn main:app --reload
   ```

6. **Откройте веб-интерфейс:**
   [http://localhost:8000/](http://localhost:8000/)

---

### 🐧 Linux

1. **Установите Python 3.8+ и git**
   ```bash
   sudo apt update
   sudo apt install python3 python3-venv python3-pip git
   ```

2. **Склонируйте репозиторий и перейдите в папку проекта:**
   ```bash
   git clone https://github.com/GameMoR1/whisper_API_que.git
   cd whisper_API_que
   ```

3. **Создайте и активируйте виртуальное окружение:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Запустите сервер:**
   ```bash
   uvicorn main:app --reload
   ```

6. **Откройте веб-интерфейс:**
   [http://localhost:8000/](http://localhost:8000/)

---

## 📚 Описание

Этот API позволяет асинхронно транскрибировать аудиофайлы с помощью Whisper на нескольких GPU, с автоматической загрузкой моделей, очередью задач и опциональным улучшением текста через GPT.  
Веб-интерфейс предоставляет мониторинг задач, GPU, логов и статуса загрузки моделей.

---

## 🗂️ Структура проекта

app/

├── main.py

├── core/

│ ├── task.py

│ ├── queue.py

│ ├── gpu.py

│ ├── model_manager.py

│ └── utils.py

├── services/

│ └── transcriber.py

├── static/

│ └── style.css

├── templates/

│ └── index.html


---

## 🛠️ Основные возможности

- **Асинхронная очередь задач** с поддержкой нескольких GPU
- **Автоматическая загрузка моделей** Whisper (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) на каждый GPU
- **Улучшение транскрипта через GPT** (g4f/OpenAI)
- **Веб-интерфейс** с мониторингом очереди, GPU, логов и статуса загрузки моделей
- **REST API** для интеграции с n8n и другими системами


---

## 🖥️ Веб-интерфейс

- Загрузка аудиофайла для транскрибации
- Чекбокс "Разделить на дорожки" — позволяет указать количество ролей и их названия (split_roads)
- Если выбран split_roads, результат будет автоматически собран в диалог по ролям с помощью GPT
- Отображение результата прямо на странице
- Мониторинг очереди, статусов задач, GPU и логов
- Красивый тёмный дизайн, автообновление данных

---

## 📝 Примеры интеграции

### Интеграция с n8n

1. **HTTP Request node**  
   - Метод: `POST`
   - URL: `http://<your_host>:8000/api/transcribe`
   - Send Binary Data: `true`
   - Binary Property: `file`
   - Add Form Fields: `model_name`, `upgrade_transcribation`, `up_speed` и др.

2. **HTTP Request node**  
   - Метод: `GET`
   - URL: `http://<your_host>:8000/api/status/{{$json["task_id"]}}`
   - Polling до статуса `done`

---


## ⚙️ Технические детали

- **split_roads**: аудиофайл делится на N дорожек (каналов), каждая дорожка транскрибируется отдельно, результат объединяется через GPT
- **GPT (g4f)**: объединяет реплики по ролям в диалог, сохраняет таймкоды и роли, форматирует текст для лучшей читабельности
- **Очередь реализована в памяти процесса**
- **Воркеры для каждого GPU**
- **Автоматическая очистка старых файлов и задач**
- **Асинхронная архитектура — задачи не блокируют друг друга**
- **Безопасность:** API публичный, без авторизации (рекомендуется использовать только во внутренней сети)

---

## 🛡️ Важно

> **Внимание:**  
> API не имеет встроенной авторизации, не используйте в открытом интернете!  
> Для продакшена рекомендуется добавить защиту и ограничение доступа.
