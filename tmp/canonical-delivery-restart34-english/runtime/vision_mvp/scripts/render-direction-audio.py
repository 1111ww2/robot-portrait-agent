"""一次性生成固定的方向提示音，平时只需播放 WAV 文件。默认打印计划，不导入语音合成或机器人库。
显式构建时只读取指定的教师音色模型，使用项目内隔离的 Piper 依赖，不打开麦克风或扬声器。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import wave


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('C:/Users/13715/OneDrive/Desktop/VOICE_AGENT-20261103/VOICE_AGENT/models/en_US-ljspeech-high.onnx')
MODEL_SHA256 = '5d4f08ba6a2a48c44592eed3ce56bf85e9de3dd4e20df90541ae68a8310c029a'
CONFIG_SHA256 = '7e1f4634af596d83cca997fb7a931ba80b70f8a316a2655ee69c55365e0ace14'
OUTPUT = ROOT / 'vision_mvp/src/leica_tracker/cue_audio'
DEPENDENCIES = ROOT / 'tmp/iteration2-voice-build/packages/runtime'
PHRASES = {
    'MOVE_LEFT': 'A little to your left.',
    'MOVE_RIGHT': 'A little to your right.',
    'MOVE_UP': 'A little higher.',
    'MOVE_DOWN': 'A little lower.',
    'MOVE_CLOSER': 'Come a little closer.',
    'MOVE_FARTHER': 'Take a small step back.',
    'ARRIVED': "That's it. Hold still.",
}
INTERACTION_PHRASES = {
    'INTRO_HELLO': 'Hi there. Nice to meet you.',
    'WAIT_MOMENT': 'One moment.',
    'WAIT_SECOND': 'Give me a second.',
    'WAIT_FRAME': 'Let me find a good frame for you.',
    'WAIT_ANGLE': "I'm looking for your best angle.",
    'WAIT_LOOK': 'Let me take a look.',
    'WAIT_SPOT': "I'm finding a good spot for you.",
    'COUNTDOWN_THREE': 'Three.',
    'COUNTDOWN_TWO': 'Two.',
    'COUNTDOWN_ONE': 'One.',
}
SAMPLE_RATE = 22050


def workspace_path(value):
    path = Path(value).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Build dependencies and output must stay inside the workspace')
    return path


def checked_model(value):
    model = Path(value)
    config = Path(str(model) + '.json')
    if model.is_symlink() or config.is_symlink():
        raise ValueError('Use the original voice model/config, not a symbolic link')
    if hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('Voice model differs from the teacher-provided checked model')
    if hashlib.sha256(config.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError('Voice config differs from the teacher-provided checked config')
    return model.resolve(), config.resolve()


def wav_record(path, text):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) != (1, 2, SAMPLE_RATE, 'NONE'):
            raise ValueError('Expected uncompressed mono PCM16 / 22050 Hz cue audio')
        duration = audio.getnframes() / SAMPLE_RATE
        if not 0 < duration <= 5.:
            raise ValueError('Direction cue must last at most five seconds')
    data = path.read_bytes()
    return {'filename': path.name, 'text': text, 'duration_s': round(duration, 6),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def render(model_path, dependency_dir, output, *, interaction_only=False):
    model, config = checked_model(model_path)
    dependencies = workspace_path(dependency_dir)
    output = workspace_path(output)
    phrases = INTERACTION_PHRASES if interaction_only else {**PHRASES, **INTERACTION_PHRASES}
    filenames = [cue.lower() + '.wav' for cue in phrases] + ['unlock.wav', 'manifest.json']
    if any((output / name).exists() for name in filenames):
        raise FileExistsError('Preserve existing generated assets; use a new output directory')
    if not (dependencies / 'piper').is_dir() or not (dependencies / 'pathvalidate').is_dir():
        raise RuntimeError('Prepare the reviewed isolated Piper 1.8.0 build dependencies first')
    # 此路径仅包含审核过的 Piper 和 pathvalidate 安装包，不会替换现有环境中的 NumPy、ONNX Runtime 或机器人依赖。
    sys.path.insert(0, str(dependencies))
    if importlib.metadata.version('piper-tts') != '1.8.0':
        raise RuntimeError('The isolated build requires Piper 1.8.0')
    versions = {name: importlib.metadata.version(name)
                for name in ('piper-tts', 'pathvalidate', 'onnxruntime', 'numpy')}
    if versions['numpy'] != '1.26.4' or versions['onnxruntime'] != '1.31.1':
        raise RuntimeError('Use the existing checked project NumPy/ONNX Runtime build')
    os.environ['ORT_DISABLE_TELEMETRY'] = '1'
    import onnxruntime
    onnxruntime.disable_telemetry_events()
    from piper import PiperVoice
    voice = PiperVoice.load(str(model), config_path=str(config), use_cuda=False,
                            download_dir=str(ROOT / 'tmp/iteration2-voice-build/resources'))
    output.mkdir(parents=True, exist_ok=True)
    assets = {}
    for cue, text in phrases.items():
        target = output / (cue.lower() + '.wav')
        with wave.open(str(target), 'wb') as audio:
            voice.synthesize_wav(text, audio)
        assets[cue] = wav_record(target, text)
    unlock = output / 'unlock.wav'
    with wave.open(str(unlock), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(b'\0\0' * int(.08 * SAMPLE_RATE))
    assets['UNLOCK'] = wav_record(unlock, '')
    manifest = {'schema_version': 1, 'voice_id': 'en_US-ljspeech-high',
                'sample_rate_hz': SAMPLE_RATE, 'channels': 1, 'sample_width_bytes': 2,
                'model_sha256': MODEL_SHA256, 'config_sha256': CONFIG_SHA256,
                'build_dependencies': versions, 'assets': assets}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for name in ('PIPER-MODEL-CARD.txt', 'PIPER-SOURCE.txt'):
        source = model.parent / name
        (output / name).write_bytes(source.read_bytes())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--interaction-only', action='store_true')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--dependency-dir', type=Path, default=DEPENDENCIES)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output, dependencies = workspace_path(args.output), workspace_path(args.dependency_dir)
    if not args.execute:
        print(json.dumps({'execute': False, 'output': str(output), 'dependencies': str(dependencies),
                          'cues': list(INTERACTION_PHRASES if args.interaction_only else {**PHRASES, **INTERACTION_PHRASES}), 'model_copied': False,
                          'microphone_or_speaker_opened': False, 'robot_connected': False}))
        return 0
    manifest = render(args.model, dependencies, output, interaction_only=args.interaction_only)
    print(json.dumps({'rendered': len(manifest['assets']),
                      'total_bytes': sum(item['bytes'] for item in manifest['assets'].values()),
                      'model_copied': False, 'microphone_or_speaker_opened': False,
                      'robot_connected': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
