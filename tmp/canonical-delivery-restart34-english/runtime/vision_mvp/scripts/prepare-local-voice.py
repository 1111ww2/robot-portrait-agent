"""前往实验室前下载并校验本地 Whisper base 模型。"""
from leica_tracker.local_speech import MODEL_DIR
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel

if __name__=='__main__':
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    snapshot_download('Systran/faster-whisper-base',revision='ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66',local_dir=MODEL_DIR,
        allow_patterns=['config.json','model.bin','tokenizer.json','vocabulary.*'])
    WhisperModel(str(MODEL_DIR),device='cpu',compute_type='int8',local_files_only=True)
    print('Local speech model ready. Audio will be processed on this computer; no speech API key needed.')
