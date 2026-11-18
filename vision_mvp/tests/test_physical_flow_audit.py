"""用于XYZ引导快照的离线回归；永远不会打开网络设备。"""
import pytest

from leica_tracker.physical_resident_bootstrap import open_resident
from test_physical_resident_bootstrap import setup


@pytest.mark.parametrize('channel,read_number', [
    ('state', 1), ('state', 2), ('state', 3),
    ('writer', 1), ('writer', 2), ('writer', 3),
    ('reader', 1), ('reader', 2),
])
def test_xyz_bootstrap_freezes_receipts_before_validation_clock(channel, read_number):
    cfg, dependencies, objects = setup()
    clock = dependencies['clock']

    class ConcurrentReceipt:
        def __init__(self, source):
            self.source = source
            self.reads = 0

        @property
        def failure(self):
            return self.source.failure

        @property
        def latest(self):
            self.reads += 1
            if self.reads == read_number:
                # 在呼叫者之前的时钟读数和此属性访问之间，生产者会收到有效的样本。
                clock.now += .001
                value, _ = self.source.latest
                self.source.latest = value, clock.now
            return self.source.latest

        def close(self):
            self.source.close()

        def send(self, *args):
            return self.source.send(*args)

    if channel == 'state':
        factory = dependencies['state_factory']
        dependencies['state_factory'] = lambda host: ConcurrentReceipt(factory(host))
    else:
        factory = dependencies['rtde_factory']

        def rtde(host, **kwargs):
            source = factory(host, **kwargs)
            selected = bool(kwargs.get('receive_only')) == (channel == 'reader')
            return ConcurrentReceipt(source) if selected else source

        dependencies['rtde_factory'] = rtde

    session = open_resident('192.0.2.1', cfg, lambda: None, dependencies=dependencies)
    try:
        assert session.engine.state == 'HANDSHAKE'
        assert len(objects['command'].sent) == 1
    finally:
        session.close()


@pytest.mark.parametrize('offset', [-.151, .001])
def test_xyz_bootstrap_still_rejects_stale_or_truly_future_feedback(offset):
    cfg, dependencies, objects = setup()
    factory = dependencies['rtde_factory']

    def rtde(host, **kwargs):
        source = factory(host, **kwargs)
        value, _ = source.latest
        source.latest = value, dependencies['clock']() + offset
        return source

    dependencies['rtde_factory'] = rtde
    with pytest.raises(RuntimeError, match='RTDE feedback unavailable or stale'):
        open_resident('192.0.2.1', cfg, lambda: None, dependencies=dependencies)
    assert 'command' not in objects
