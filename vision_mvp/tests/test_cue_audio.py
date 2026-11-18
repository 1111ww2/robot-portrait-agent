"""小型经过认证的预录制提示；无TTS、麦克风或机器人运行时。"""
import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request
import wave

import pytest


@pytest.fixture(autouse=True)
def clear_audio_cache():
    from leica_tracker.cue_audio import read_cue
    read_cue.cache_clear()
    yield
    read_cue.cache_clear()


def cue_fixture(tmp_path, monkeypatch):
    from leica_tracker import cue_audio
    stream=io.BytesIO()
    with wave.open(stream,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(22050)
        wav.writeframes(b'\0\0'*2205)
    raw=stream.getvalue()
    (tmp_path/'move_left.wav').write_bytes(raw)
    (tmp_path/'manifest.json').write_text(json.dumps({'schema_version':1,'assets':{
        'MOVE_LEFT':{'filename':'move_left.wav','duration_s':.1,'bytes':len(raw),
                     'sha256':hashlib.sha256(raw).hexdigest()}}}),encoding='utf-8')
    monkeypatch.setattr(cue_audio,'ASSET_ROOT',tmp_path)
    cue_audio.read_cue.cache_clear()
    return raw


def test_audio_manifest_and_pcm_are_validated_before_serving(tmp_path,monkeypatch):
    from leica_tracker import cue_audio
    raw=cue_fixture(tmp_path,monkeypatch)
    assert cue_audio.read_cue('MOVE_LEFT')==raw
    for bad in ('../../.env','move_left','MOVE_LEFT.wav','UNKNOWN',None):
        with pytest.raises(ValueError):cue_audio.read_cue(bad)
    cue_audio.read_cue.cache_clear()
    (tmp_path/'move_left.wav').write_bytes(raw[:-8])
    with pytest.raises(ValueError):cue_audio.read_cue('MOVE_LEFT')


def test_audio_route_requires_pairing_and_an_exact_allowlisted_cue(tmp_path,monkeypatch):
    from leica_tracker.guest_display import GuestDisplay
    raw=cue_fixture(tmp_path,monkeypatch)
    with GuestDisplay(port=0) as guest:
        def get(query,token=None):
            return urllib.request.urlopen(urllib.request.Request(guest.origin+'/cue-audio'+query,
                headers={'X-Guest-Token':token or guest.token}),timeout=2)
        with get('?cue=MOVE_LEFT') as response:
            assert response.status==200 and response.headers.get_content_type()=='audio/wav'
            assert response.read()==raw
        for query,token,status in (('?cue=MOVE_LEFT','wrong',403),('',None,400),
             ('?cue=MOVE_LEFT&cue=MOVE_RIGHT',None,400),('?cue=MOVE_LEFT&path=.env',None,400),
             ('?cue=../../.env',None,404)):
            with pytest.raises(urllib.error.HTTPError) as exc:get(query,token)
            assert exc.value.code==status


def test_paired_https_forwards_cue_audio_without_exposing_extra_routes(tmp_path,monkeypatch):
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.guest_https import GuestHTTPS,create_certificates
    raw=cue_fixture(tmp_path,monkeypatch)
    files=create_certificates(tmp_path/'certs','127.0.0.1')
    context=ssl.create_default_context(cafile=str(files['ca']))
    with GuestDisplay(port=0) as guest, GuestHTTPS(files,upstream_port=guest.port,
            host='127.0.0.1',port=0,setup_port=0,addresses=['127.0.0.1']) as proxy:
        url=f'https://127.0.0.1:{proxy.port}/cue-audio?cue=MOVE_LEFT'
        req=urllib.request.Request(url,headers={'X-Guest-Token':guest.token})
        with urllib.request.urlopen(req,context=context,timeout=3) as response:
            assert response.headers.get_content_type()=='audio/wav' and response.read()==raw
        req=urllib.request.Request(url,headers={'X-Guest-Token':'wrong'})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req,context=context,timeout=3)
        assert exc.value.code==403
