"""iPad 只播放允许列表内的预录语音，运行时不合成语音或访问录音设备。"""
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import wave


ASSET_ROOT=Path(__file__).with_name('cue_audio')
CUE_FILES={cue:cue.lower()+'.wav' for cue in (
    'MOVE_LEFT','MOVE_RIGHT','MOVE_UP','MOVE_DOWN','MOVE_CLOSER','MOVE_FARTHER','ARRIVED','UNLOCK',
    'INTRO_HELLO','WAIT_MOMENT','WAIT_SECOND','WAIT_FRAME','WAIT_ANGLE','WAIT_LOOK','WAIT_SPOT',
    'COUNTDOWN_THREE','COUNTDOWN_TWO','COUNTDOWN_ONE')}
MAX_BYTES=512*1024


@lru_cache(maxsize=32)
def read_cue(cue):
    """只读取随应用发布、名称获准且哈希匹配的 PCM 音频片段。"""
    if not isinstance(cue,str) or cue not in CUE_FILES:
        raise ValueError('Unknown direction audio cue')
    root=ASSET_ROOT.resolve()
    manifest_path=root/'manifest.json'
    if manifest_path.is_symlink():raise ValueError('Invalid audio manifest')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    item=manifest.get('assets',{}).get(cue)
    if (manifest.get('schema_version')!=1 or not isinstance(item,dict)
            or item.get('filename')!=CUE_FILES[cue]):
        raise ValueError('Invalid audio asset manifest')
    path=root/CUE_FILES[cue]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Invalid audio asset path')
    raw=path.read_bytes()
    if (not 44<=len(raw)<=MAX_BYTES or item.get('bytes')!=len(raw)
            or item.get('sha256')!=hashlib.sha256(raw).hexdigest()):
        raise ValueError('Audio asset integrity mismatch')
    try:
        with wave.open(io.BytesIO(raw),'rb') as wav:
            if (wav.getnchannels()!=1 or wav.getsampwidth()!=2
                    or wav.getframerate()!=22050 or wav.getcomptype()!='NONE'):
                raise ValueError('Unsupported direction audio format')
            frames=wav.getnframes()
            if not 0<frames<=22050*8 or len(wav.readframes(frames+1))!=frames*2:
                raise ValueError('Invalid direction audio length')
            duration=item.get('duration_s')
            if (type(duration) not in (int,float) or not math.isfinite(duration)
                    or abs(duration-frames/22050)>.001):
                raise ValueError('Invalid direction audio duration')
    except (wave.Error,EOFError) as exc:
        raise ValueError('Invalid direction audio file') from exc
    return raw
