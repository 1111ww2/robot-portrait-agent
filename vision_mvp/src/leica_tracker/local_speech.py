"""在本地执行有限时长的语音识别。音频只存在内存中，不进入机器人进程的标准输出。"""
import io
from pathlib import Path
import threading
import numpy as np

MODEL_DIR=Path(__file__).resolve().parents[2]/'models'/'speech-base'
MAX_AUDIO_BYTES=2*1024*1024
MAX_AUDIO_SECONDS=20


class LocalSpeech:
    def __init__(self,model_dir=MODEL_DIR):
        self.model_dir=Path(model_dir);self._model=None;self._lock=threading.Lock()

    @property
    def ready(self):
        return self._model is not None or (self.model_dir/'model.bin').is_file()

    @staticmethod
    def decode(data):
        if not 1<=len(data)<=MAX_AUDIO_BYTES:raise ValueError('Audio size invalid')
        import av
        try:
            chunks=[];count=0
            with av.open(io.BytesIO(data)) as container:
                resampler=av.AudioResampler(format='s16',layout='mono',rate=16000)
                for frame in container.decode(audio=0):
                    for part in resampler.resample(frame):
                        count+=part.samples
                        if count>MAX_AUDIO_SECONDS*16000:raise ValueError('Recording too long')
                        chunks.append(part.to_ndarray().flatten())
                for part in resampler.resample(None):
                    count+=part.samples
                    if count>MAX_AUDIO_SECONDS*16000:raise ValueError('Recording too long')
                    chunks.append(part.to_ndarray().flatten())
            if not chunks or count<1600:raise ValueError('Recording too short')
            return np.concatenate(chunks).astype(np.float32)/32768.
        except Exception as error:
            raise ValueError('Audio could not be decoded within recording limits') from error

    def transcribe(self,data,language='en'):
        if language not in ('en','zh'):raise ValueError('Unsupported speech language')
        if not self._lock.acquire(blocking=False):raise BlockingIOError('Speech recognition busy')
        try:
            if not self.ready:raise RuntimeError('Run prepare-local-voice.py on the laptop')
            audio=self.decode(data)
            if self._model is None:
                from faster_whisper import WhisperModel
                self._model=WhisperModel(str(self.model_dir),device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
            segments,_=self._model.transcribe(audio,language=language,beam_size=1,
                vad_filter=True,condition_on_previous_text=False)
            text=' '.join(s.text.strip() for s in segments).strip()
            return text[:300]
        finally:self._lock.release()
