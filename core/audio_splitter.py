import os
import tempfile
import subprocess
from typing import List

def split_audio_channels(input_path: str, num_channels: int, role_names: List[str], temp_dir: str) -> List[str]:
    """
    Делит аудиофайл на отдельные дорожки по каналам с помощью ffmpeg pan.
    Возвращает список путей к файлам дорожек (в порядке role_names).
    """
    output_paths = []
    for i, role in enumerate(role_names):
        ext = os.path.splitext(input_path)[-1]
        fd, out_path = tempfile.mkstemp(suffix=f"_{role}{ext}", dir=temp_dir)
        os.close(fd)
        # ffmpeg -i input.wav -filter_complex "[0:a]pan=mono|c0=c{i}[out]" -map "[out]" out_role.wav
        pan_expr = f"mono|c0=c{i}"
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter_complex", f"[0:a]pan={pan_expr}[out]", "-map", "[out]", out_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg pan error: {e.stderr.decode(errors='ignore')}")
        output_paths.append(out_path)
    return output_paths
