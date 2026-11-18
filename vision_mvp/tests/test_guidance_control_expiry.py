"""在内存中模拟 HTTP 和设备，检查实体控制所有者到期的处理，不打开套接字。"""
from types import SimpleNamespace

import pytest

from leica_tracker import guest_display


@pytest.fixture
def encounter(monkeypatch):
    now = [10.]
    monkeypatch.setattr(guest_display, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(guest_display, 'ThreadingHTTPServer', lambda *args: SimpleNamespace(
        server_address=('127.0.0.1', 8767), serve_forever=lambda: None))
    capture = SimpleNamespace(stream_fresh=lambda: True, fresh=lambda: True,
                              connected=lambda: True)
    display = guest_display.GuestDisplay(remote_capture=capture, physical_feedback=True)
    display._set_action_locked('start', 'owner', control_protocol=1)
    identity = {'instance': display._control_instance, 'epoch': display._control_epoch}
    registered = display._physical_feedback_update({'op': 'register', 'mode': 'intro', **identity})
    backend = {**identity, **registered}
    display._physical_feedback_update({'op': 'intro', 'intro_id': 10_001, 'status': 'active', **backend})
    display._physical_feedback_update({'op': 'intro', 'intro_id': 10_001, 'status': 'ready', **backend})
    now[0] = 10.15
    display._physical_feedback_update({'op': 'heartbeat', **backend})
    return display, now, identity, backend


def test_guidance_owner_expiry_revokes_even_with_live_camera_and_backend(encounter):
    display, now, _, _ = encounter
    now[0] = 10.312
    assert display.control_snapshot()['control_fresh'] is False
    assert display.physical_backend_available() is False
    assert display._physical_revocation['reason'] == 'control_heartbeat_expired'


def test_delayed_owner_heartbeat_cannot_revive_expired_guidance(encounter):
    display, now, identity, _ = encounter
    now[0] = 10.312
    with pytest.raises(ValueError, match='expired'):
        display._control_heartbeat_locked({**identity, 'sequence': 1}, 'owner', now[0])
    assert display._seen == 10.
    assert not display.physical_backend_available()


def test_backend_heartbeat_cannot_keep_guidance_after_owner_expiry(encounter):
    display, now, _, backend = encounter
    now[0] = 10.312
    with pytest.raises(ValueError, match='expired'):
        display._physical_feedback_update({'op': 'heartbeat', **backend})


def test_new_target_cannot_register_with_expired_owner_control(encounter):
    display, now, _, _ = encounter
    display._set_action_locked('restart', 'owner', control_protocol=1)
    identity = {'instance': display._control_instance, 'epoch': display._control_epoch}
    now[0] += .312
    with pytest.raises(ValueError, match='inactive'):
        display._physical_feedback_update({'op': 'register', 'mode': 'guidance', **identity})


def test_fresh_owner_keeps_guidance_available(encounter):
    display, now, identity, _ = encounter
    now[0] = 10.29
    assert display._control_heartbeat_locked({**identity, 'sequence': 1}, 'owner', now[0])['accepted']
    now[0] = 10.4
    assert display.physical_backend_available()
