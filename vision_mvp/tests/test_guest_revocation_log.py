"""验证首次故障原因的诊断输出，不启动访客服务器。"""
import json
import threading
from types import SimpleNamespace

from leica_tracker import guest_display


def guest():
    value = object.__new__(guest_display.GuestDisplay)
    value._physical_revocation = None
    value._control_epoch = 7
    value._physical_backend_mode = 'guidance'
    value._seen = 10.25
    value._physical_backend_at = 10.3
    value.remote_capture = SimpleNamespace(_condition=threading.Condition(), _last_received=10.4)
    value.token = 'private-pairing-token-do-not-log'
    value._control_instance = 'private-control-instance-do-not-log'
    return value


def records(stderr):
    prefix = 'PHYSICAL_CONTROL_REVOKED '
    return [json.loads(line[len(prefix):]) for line in stderr.splitlines() if line.startswith(prefix)]


def test_first_revocation_is_logged_once_and_later_cleanup_preserves_it(capsys):
    value = guest()
    value._record_physical_revocation('control_heartbeat_expired', 10.5)
    expected = dict(value._physical_revocation)
    value._record_physical_revocation('backend_fault', 11.)
    captured = capsys.readouterr()
    assert records(captured.err) == [expected]
    assert captured.out == '' and value._physical_revocation == expected
    assert expected == {'reason':'control_heartbeat_expired','epoch':7,'mode':'guidance',
                        'control_age_s':.25,'backend_age_s':.2,'camera_age_s':.1}
    assert value.token not in captured.err and value._control_instance not in captured.err
    assert set(expected) == {'reason','epoch','mode','control_age_s','backend_age_s','camera_age_s'}


def test_new_epoch_registration_can_emit_its_own_first_cause(capsys, monkeypatch):
    value = guest()
    value._record_physical_revocation('camera_stale', 10.75)
    value._control_epoch = 8
    value._physical_feedback_enabled = True
    value._active = True; value._closed = False
    value._physical_registered_epoch = None
    value._expire = lambda:None
    value._camera_stream_fresh = lambda:True
    value._control_stream_fresh = lambda now:True
    monkeypatch.setattr(guest_display, 'time', SimpleNamespace(monotonic=lambda:11.))
    result = value._physical_feedback_update({'op':'register','instance':value._control_instance,'epoch':8})
    assert value._physical_revocation is None
    value._record_physical_revocation('backend_heartbeat_expired', 11.5)
    logged = records(capsys.readouterr().err)
    assert [record['epoch'] for record in logged] == [7,8]
    assert [record['reason'] for record in logged] == ['camera_stale','backend_heartbeat_expired']
    assert result['lease'] not in json.dumps(logged)
