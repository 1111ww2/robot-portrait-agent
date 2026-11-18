import io
import wave
import numpy as np
import pytest
from leica_tracker.local_speech import LocalSpeech

pytest.importorskip("av",reason="Install handoff extra for local voice")


def wav(seconds=1):
    out=io.BytesIO()
    with wave.open(out,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(np.zeros(round(seconds*16000),dtype='<i2').tobytes())
    return out.getvalue()


def test_audio_duration_and_malformed_input_are_bounded():
    assert LocalSpeech.decode(wav()).shape==(16000,)
    with pytest.raises(ValueError):LocalSpeech.decode(wav(21))
    with pytest.raises(ValueError):LocalSpeech.decode(b'not audio')


def test_transcription_is_local_bounded_and_serialized(tmp_path):
    from types import SimpleNamespace
    calls=[]
    class Model:
        def transcribe(self,audio,**kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text=' Hello, can you hear me?')]),None
    speech=LocalSpeech(model_dir=tmp_path);speech._model=Model()
    assert speech.transcribe(wav(),'en')=='Hello, can you hear me?'
    assert calls[0]['language']=='en'
    speech._lock.acquire()
    try:
        with pytest.raises(BlockingIOError):speech.transcribe(wav(),'en')
    finally:speech._lock.release()
