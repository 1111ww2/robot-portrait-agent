"""下一次开始得到接纳前，参考位 HTTP 工作线程必须退出。"""
import threading
from types import SimpleNamespace

import pytest

from leica_tracker import physical_interaction_supervisor as module
from test_physical_interaction_supervisor import setup


def test_reference_worker_close_waits_for_thread_and_fails_on_bounded_timeout():
    entered = threading.Event()
    release = threading.Event()
    def operation():
        entered.set()
        release.wait(2.)
    worker = module._Worker(operation)
    assert entered.wait(.5)
    try:
        with pytest.raises(RuntimeError, match='worker'):
            worker.close(.01)
        assert not worker.done
    finally:
        release.set()
    worker.close(.5)
    assert worker.done and not worker._thread.is_alive()


def test_supervisor_workers_share_one_deadline_and_deduplicate(monkeypatch):
    *_, sup = setup()
    clock = [10.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []
    class Worker:
        done = False
        def close(self, timeout):
            calls.append(timeout)
            clock[0] += .3
            self.done = True
    sup._job = Worker()
    sup._revocation = Worker()
    sup.close_workers(.5)
    assert calls == pytest.approx([.5, .2])
    calls.clear()
    sup._job = sup._revocation = Worker()
    sup.close_workers(.5)
    assert len(calls) == 1


def test_unknown_pending_worker_blocks_future_start_cleanup():
    *_, sup = setup()
    sup._job = SimpleNamespace(done=False)
    with pytest.raises(RuntimeError, match='worker'):
        sup.close_workers(.01)


def test_immediate_completed_fake_worker_is_already_retired():
    *_, sup = setup()
    sup._job = SimpleNamespace(done=True)
    sup._revocation = False
    sup.close_workers(.01)


@pytest.mark.parametrize('state,kind', [
    ('SAMPLE_PENDING', 'sample'), ('PHOTO_LOOKUP', 'photo_source'),
    ('PHOTO_PENDING', 'photo_ready')])
def test_network_resume_cannot_silently_drop_a_pending_reference_worker(state, kind):
    *_, sup = setup()
    sup.state = state
    sup._job_kind = kind
    sup._job_started = 1.
    sup._job = SimpleNamespace(done=False)
    sup.pause_for_network(1.1)
    sup.resume_after_network(1.2, 1_200_000_000)
    assert sup._job is None
    with pytest.raises(RuntimeError, match='worker'):
        sup.close_workers(.01)


def test_detached_reference_workers_share_cleanup_budget_and_are_deduplicated(monkeypatch):
    *_, sup = setup()
    clock = [10.]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    calls = []
    class Worker:
        done = False
        def close(self, timeout):
            calls.append(timeout)
            clock[0] += .3
            self.done = True
    detached = Worker()
    sup._job = detached
    sup._job_kind = 'sample'
    sup.state = 'SAMPLE_PENDING'
    sup.pause_for_network(1.1)
    sup.resume_after_network(1.2, 1_200_000_000)
    sup._job = Worker()
    sup._revocation = detached
    sup.close_workers(.5)
    assert calls == pytest.approx([.5, .2])
