"""使用模拟通道测试会话过期后的界面恢复，不连接机器人。"""
import pytest

from test_physical_runtime import runtime


@pytest.mark.parametrize('state', ('SETTLING', 'SAMPLE_PENDING', 'RELEASE_PENDING'))
def test_expired_unstarted_direction_clears_lights_without_ending_session(state):
    run, session, supervisor, publisher = runtime()
    control = dict(run.control)

    def accept(event, *, now):
        supervisor.state = 'BEGIN_PENDING'
        return True

    supervisor.accept_event = accept
    event = {'event': 'person_guidance', 'cue': 'MOVE_LEFT',
             'source_monotonic_ns': 10_000_000_000,
             'control_instance': 'x', 'control_epoch': 1}
    run.step(event, now=10.)
    action_id = run.action['action_id']
    supervisor.recovering_expiry = True
    supervisor.state = state
    session.writer.latest = (session.writer.latest[0], 10.02)
    session.reader.latest = (session.reader.latest[0], 10.02)

    run.step(None, now=10.02)

    assert run.action is None
    assert [p for p in publisher.calls if p.get('op') == 'clear'] == [
        {'op': 'clear', 'action_id': action_id}]
    assert not any(p.get('op') == 'sample' for p in publisher.calls)
    assert run.control == control
    assert session.gate.can_send
    assert supervisor.state == state

    # 持续静止/参考验证不得重新发布提示。
    session.writer.latest = (session.writer.latest[0], 10.04)
    session.reader.latest = (session.reader.latest[0], 10.04)
    run.step(None, now=10.04)
    assert len([p for p in publisher.calls if p.get('op') == 'clear']) == 1
    assert not any(p.get('op') == 'sample' for p in publisher.calls)
