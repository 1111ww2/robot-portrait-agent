"""仅使用假货进行练习预览CLI布线；没有机器人或过程操作。"""
import importlib.util
import io
import json
import tempfile
import threading
from pathlib import Path

import pytest

from leica_tracker.physical_step_preview import StepSequence
from test_physical_step_preview import example


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('preview_entry', ROOT / 'vision_mvp/scripts/run-physical-preview.py')
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_operator_stop_during_asynchronous_open_never_ticks_any_step():
    draft, step = example()
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    calls = []
    class Session:
        def tick(self): calls.append('tick'); return 'ACTIVE'
        def stop(self): calls.append('stop')
        def close(self): calls.append('close'); closed.set()
    def opener(host,profile,reader):
        entered.set()
        assert release.wait(2)
        assert reader()['active'] is False
        return Session()
    preview = StepSequence(draft,'192.0.2.1',lambda:{'active':True},opener=opener)
    class Input:
        def __iter__(self):
            yield 'next\n'
            assert entered.wait(2)
            yield 'stop\n'
            assert preview._cancel.wait(2)
            release.set()
            assert closed.wait(2)
            yield 'close\n'
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
        directory=Path(tmp); (directory/'01.json').write_text(json.dumps(step),encoding='utf-8')
        ENTRY.interactive(preview,directory,directory/'report.json',stdin=Input(),output=io.StringIO())
    assert 'tick' not in calls and calls == ['stop','close']
    assert preview.state == 'STOPPED'


def test_guest_process_exit_stops_preview_without_starting_a_step():
    draft,_=example(); calls=[]
    preview=StepSequence(draft,'192.0.2.1',lambda:None,
                         opener=lambda *args: calls.append('unexpected'))
    guest=type('Guest',(),{'poll':lambda self:1})()
    with pytest.raises(RuntimeError,match='guest exited'):
        ENTRY.interactive(preview,ROOT/'tmp',ROOT/'tmp/report-unused.json',stdin=io.StringIO(),
                          output=io.StringIO(),guest_process=guest)
    assert preview.state=='STOPPED' and calls==[]


def test_default_entry_lists_unapproved_points_without_starting_any_process(monkeypatch,capsys):
    draft,_=example()
    monkeypatch.setattr(ENTRY.subprocess,'Popen',lambda *a,**k:pytest.fail('offline listing must not start a process'))
    (ROOT/'tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
        path=Path(tmp)/'candidate.json';path.write_text(json.dumps(draft),encoding='utf-8')
        assert ENTRY.main(['--draft',str(path)])==0
    items=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [x['point'] for x in items]==list(range(1,15))
    assert all(x['status']=='DRAFT' for x in items)
