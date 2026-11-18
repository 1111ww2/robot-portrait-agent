"""无需真实凭证或网络摄像头即可进行便携式入职检查。"""
import json
from pathlib import Path
import runpy
import sys

import pytest


def copied_script(tmp_path,name):
    root=tmp_path/'new laptop with spaces'
    target=root/'vision_mvp/scripts'/name;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((Path(__file__).parents[1]/'scripts'/name).read_bytes())
    return root,runpy.run_path(str(target))


def test_local_hidden_key_setup_writes_only_ignored_file_and_refuses_overwrite(tmp_path,monkeypatch,capsys):
    root,mod=copied_script(tmp_path,'configure-gemini.py')
    monkeypatch.setattr(sys,'argv',['configure-gemini.py'])
    monkeypatch.setattr(mod['getpass'],'getpass',lambda prompt:'local-test-value')
    mod['main']()
    assert (root/'.env.local').read_text()=='GEMINI_API_KEY=local-test-value\n'
    assert 'local-test-value' not in capsys.readouterr().out
    with pytest.raises(SystemExit,match='already exists'):mod['main']()
    assert (root/'.env.local').read_text()=='GEMINI_API_KEY=local-test-value\n'


def test_pairing_qr_is_created_locally_for_lan_not_loopback_without_echoing_token(tmp_path,monkeypatch,capsys):
    pytest.importorskip('qrcode')
    from leica_tracker.guest_display import GuestDisplay
    root,mod=copied_script(tmp_path,'pair-ipad.py')
    with GuestDisplay(port=0) as guest:
        link=root/'tmp/guidance-live/guest-link.json';link.parent.mkdir(parents=True)
        link.write_text(json.dumps(dict(port=guest.port,key=guest.token)))
        monkeypatch.setattr(sys,'argv',['pair-ipad.py','--host','192.168.20.9','--no-open'])
        mod['main']()
        page=link.with_name('ipad-pair.html').read_text()
        assert '<svg' in page and f'http://192.168.20.9:{guest.port}/?key=' in page
        assert guest.token not in capsys.readouterr().out
        monkeypatch.setattr(sys,'argv',['pair-ipad.py','--host','127.0.0.1','--no-open'])
        with pytest.raises(SystemExit,match='LAN IPv4'):mod['main']()


def test_missing_bootstrap_tools_stop_before_dependency_install(tmp_path,monkeypatch):
    _,mod=copied_script(tmp_path,'bootstrap-laptop.py')
    monkeypatch.setattr(sys,'argv',['bootstrap-laptop.py'])
    monkeypatch.setattr(mod['shutil'],'which',lambda name:None)
    monkeypatch.setattr(mod['subprocess'],'run',lambda *a,**kw:pytest.fail('installation attempted with missing tools'))
    with pytest.raises(SystemExit,match='Missing uv'):mod['main']()


def test_key_setup_refuses_echo_fallback_when_no_interactive_terminal(tmp_path,monkeypatch):
    root,mod=copied_script(tmp_path,'configure-gemini.py')
    monkeypatch.setattr(sys,'argv',['configure-gemini.py'])
    def no_tty(prompt):raise mod['getpass'].GetPassWarning('no terminal')
    monkeypatch.setattr(mod['getpass'],'getpass',no_tty)
    with pytest.raises(SystemExit,match='No hidden-input terminal'):
        mod['main']()
    assert not (root/'.env.local').exists()
