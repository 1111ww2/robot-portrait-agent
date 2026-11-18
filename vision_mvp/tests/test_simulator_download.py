"""中断或损坏的下载不能被当成可信缓存。"""
import hashlib
import io
from pathlib import Path
import runpy

import pytest


def installer():
    mod=runpy.run_path(str(Path(__file__).parents[1]/'scripts/setup-teacher-simulator.py'))
    return mod['download_archive']


def test_public_download_uses_explicit_client_and_checks_before_cache(tmp_path,monkeypatch):
    download=installer();payload=b'fixture-publisher-bytes';calls=[]
    monkeypatch.setitem(download.__globals__,'SHA256',hashlib.sha256(payload).hexdigest())
    def response(request,timeout):
        calls.append(request);return io.BytesIO(payload)
    monkeypatch.setattr(download.__globals__['urllib'].request,'urlopen',response)
    target=tmp_path/'source.tar.gz';download(target)
    assert target.read_bytes()==payload
    assert calls[0].get_header('User-agent')=='Mozilla/5.0'
    assert not target.with_suffix('.gz.part').exists()


def test_corrupt_download_preserves_existing_cache_and_removes_partial(tmp_path,monkeypatch):
    download=installer();target=tmp_path/'source.tar.gz';target.write_bytes(b'existing-cache')
    monkeypatch.setattr(download.__globals__['urllib'].request,'urlopen',lambda *a,**kw:io.BytesIO(b'corrupt'))
    with pytest.raises(SystemExit,match='checksum mismatch'):download(target)
    assert target.read_bytes()==b'existing-cache'
    assert not target.with_suffix('.gz.part').exists()
