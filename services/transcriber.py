import torch
import subprocess
import tempfile
import os
import whisper
import g4f # pip install g4f
import time

def build_atempo_filters(up_speed):
    filters = []
    speed = up_speed
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed *= 2.0
    filters.append(f"atempo={speed:.2f}")
    return ",".join(filters)

def improve_transcription_with_gpt(transcript_text):
    gpt_prompt = (
        "Вот расшифровка диалога между двумя спикерами: сотрудником и клиентом. "
        "Раздели текст по репликам спикеров, подпиши кто говорит (Сотрудник или Клиент), "
        "исправь явные ошибки и сделай текст более читабельным. "
        "Сохрани тайминги в формате [mm:ss] перед каждой репликой. ЕСЛИ разговора нет, а был автоответчик или что-то другое, "
        "в первой строке напиши *false*, в другом случае *true*\n\n"
        f"{transcript_text}"
    )
    improved_text = g4f.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": gpt_prompt}],
    ).strip()
    return improved_text

def transcribe_audio(task, gpu_id):
    torch.cuda.set_device(gpu_id)
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    atempo_filter = build_atempo_filters(task.up_speed)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", task.file_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2",
        "-filter:a", atempo_filter,
        mp3_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.remove(mp3_path)
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
    try:
        # Модель загружается только на время задачи!
        model = whisper.load_model(task.model_name, device=f"cuda:{gpu_id}")
        transcribe_kwargs = {
            "fp16": True,
            "beam_size": 5
        }
        if task.initial_prompt:
            transcribe_kwargs["initial_prompt"] = task.initial_prompt

        # Засекаем время выполнения транскрибации
        start_time = time.time()
        result = model.transcribe(mp3_path, **transcribe_kwargs)
        end_time = time.time()
        processing_time = end_time - start_time
    finally:
        # Удаляем модель и очищаем VRAM
        del model
        torch.cuda.empty_cache()
        os.remove(mp3_path)

    def format_timestamp(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    formatted_text = "\n".join(
        f"[{format_timestamp(seg['start'])}] {seg['text'].strip()}"
        for seg in result['segments']
    )

    # Улучшение через GPT, если требуется
    if getattr(task, "upgrade_transcribation", False):
        try:
            improved_text = improve_transcription_with_gpt(formatted_text)
            return improved_text, processing_time
        except Exception as e:
            return formatted_text + f"\n\n[Ошибка улучшения через GPT: {e}]", processing_time
    else:
        return formatted_text, processing_time
