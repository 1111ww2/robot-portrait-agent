"""规范的导出必须包括固定线索，同时排除无关的JSON。"""
from pathlib import Path
import pytest


def test_only_fixed_audio_assets_and_manifest_join_runtime_identity(tmp_path):
    from leica_tracker.canonical_delivery import _runtime_sources,runtime_source_identity
    package=tmp_path/'vision_mvp/src/leica_tracker';assets=package/'cue_audio'
    assets.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (assets/'move_left.wav').write_bytes(b'first clip')
    (assets/'unlock.wav').write_bytes(b'silent clip')
    (assets/'intro_hello.wav').write_bytes(b'greeting')
    (assets/'wait_frame.wav').write_bytes(b'framing wait')
    (assets/'countdown_three.wav').write_bytes(b'number three')
    (assets/'manifest.json').write_text('{"schema_version":1}')
    (assets/'private.json').write_text('{"token":"not an asset"}')
    (assets/'unreviewed.wav').write_bytes(b'not an asset')
    (assets/'README.md').write_text('source attribution')
    (assets/'PIPER-SOURCE.txt').write_text('runtime and model provenance')
    (assets/'PIPER-MODEL-CARD.txt').write_text('provided voice model card')
    names={p.name for p in _runtime_sources(tmp_path)}
    assert {'move_left.wav','unlock.wav','intro_hello.wav','wait_frame.wav','countdown_three.wav','manifest.json','README.md',
            'PIPER-SOURCE.txt','PIPER-MODEL-CARD.txt'}<=names
    assert 'private.json' not in names and 'unreviewed.wav' not in names
    first=runtime_source_identity(tmp_path)['shared_runtime_source_sha256']
    (assets/'move_left.wav').write_bytes(b'changed clip')
    assert runtime_source_identity(tmp_path)['shared_runtime_source_sha256']!=first
