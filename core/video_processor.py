import os
import tempfile
import subprocess
from typing import Optional


def extract_audio_from_video(video_path: str, temp_dir: str) -> str:
    """
    Извлекает аудио из видеофайла с помощью ffmpeg.
    
    Args:
        video_path: Путь к видеофайлу
        temp_dir: Директория для временных файлов
        
    Returns:
        Путь к извлеченному аудиофайлу
        
    Raises:
        RuntimeError: Если ffmpeg не может извлечь аудио
    """
    # Создаем временный файл для аудио
    fd, audio_path = tempfile.mkstemp(suffix='.wav', dir=temp_dir)
    os.close(fd)
    
    # Команда ffmpeg для извлечения аудио
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn',  # Без видео
        '-acodec', 'pcm_s16le',  # Кодек для WAV
        '-ar', '44100',  # Частота дискретизации
        '-ac', '2',  # Стерео
        audio_path
    ]
    
    try:
        result = subprocess.run(
            ffmpeg_cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return audio_path
    except subprocess.CalledProcessError as e:
        # Удаляем временный файл в случае ошибки
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise RuntimeError(f"Ошибка извлечения аудио из видео: {e.stderr}")


def is_video_file(file_path: str) -> bool:
    """
    Проверяет, является ли файл видеофайлом по расширению.
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        True если файл является видеофайлом
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in video_extensions


def get_video_info(video_path: str) -> dict:
    """
    Получает информацию о видеофайле с помощью ffprobe.
    
    Args:
        video_path: Путь к видеофайлу
        
    Returns:
        Словарь с информацией о видео (длительность, аудио дорожки и т.д.)
    """
    ffprobe_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        'format=duration,size:stream=codec_type,channels,sample_rate',
        '-of', 'json', video_path
    ]
    
    try:
        result = subprocess.run(
            ffprobe_cmd, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        import json
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        return {"error": f"Не удалось получить информацию о видео: {e}"}

