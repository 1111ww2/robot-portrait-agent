"""工作线程首次收到确认和监管器随后轮询是两个时刻。此测试不做输入输出。"""
import pytest
from leica_tracker.physical_resident_engine import Installation,ResidentEngine
from test_physical_resident_engine import reviewed,Gate,feedback


def pending():
    cfg=Installation(reviewed());gate=Gate();engine=ResidentEngine(cfg,gate,123)
    engine.tick(feedback(cfg,123),sampled_at=1.,now=1.)
    return cfg,gate,engine


def test_on_time_worker_ack_survives_late_parent_poll_with_fresh_current_feedback():
    cfg,gate,engine=pending()
    sample=feedback(cfg,123,state=0,seq=1);sample.update(ack_sampled_at=1.05,worker_sampled_at=1.219)
    engine.tick(sample,sampled_at=1.219,now=1.219)
    assert engine.state=='READY' and not gate.stopped
    assert gate.calls[-1][:3]==(2,0,123)


@pytest.mark.parametrize('proof',[1.200001,1.219])
def test_matching_but_genuinely_late_worker_ack_still_faults(proof):
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=proof,worker_sampled_at=1.219)
    with pytest.raises(RuntimeError,match='acknowledgement timeout'):
        engine.tick(sample,sampled_at=1.219,now=1.219)
    assert gate.stopped and len(gate.calls)==1


def test_ack_at_original_200ms_boundary_is_accepted_without_an_extended_budget():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=1.2,worker_sampled_at=1.219)
    engine.tick(sample,sampled_at=1.219,now=1.219)
    assert engine.state=='READY' and not gate.stopped


@pytest.mark.parametrize('proof',[True,float('nan'),float('inf'),-.1,.99,1.22])
def test_invalid_or_pre_send_or_future_receipt_never_sends_a_new_packet(proof):
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=proof,worker_sampled_at=1.1)
    with pytest.raises(RuntimeError):engine.tick(sample,sampled_at=1.1,now=1.1)
    assert gate.stopped and len(gate.calls)==1


def test_latest_feedback_freshness_is_required_even_with_an_on_time_ack():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=1.1)
    with pytest.raises(RuntimeError,match='Stale'):
        engine.tick(sample,sampled_at=1.1,now=1.3)
    assert gate.stopped and len(gate.calls)==1


def test_missing_receipt_cannot_extend_the_original_timeout():
    cfg,gate,engine=pending()
    with pytest.raises(RuntimeError,match='acknowledgement timeout'):
        engine.tick(feedback(cfg,123,state=0,seq=1),sampled_at=1.219,now=1.219)
    assert gate.stopped and len(gate.calls)==1


def test_nonmatching_ack_cannot_borrow_a_timely_receipt():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=0)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=1.219)
    with pytest.raises(RuntimeError,match='acknowledgement timeout'):
        engine.tick(sample,sampled_at=1.219,now=1.219)
    assert gate.stopped and len(gate.calls)==1


def test_stopped_resident_cannot_resume_using_an_old_timely_ack():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=2,seq=1)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=1.219)
    # 已使能的会话停止后仍须保留最初的拒绝原因。
    engine.state='READY'
    with pytest.raises(RuntimeError,match='stopped or faulted'):
        engine.tick(sample,sampled_at=1.219,now=1.219)
    assert gate.stopped and len(gate.calls)==1


@pytest.mark.parametrize('later',[1.06,None])
def test_first_receipt_for_the_same_ack_identity_cannot_change_or_disappear(later):
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=1.08)
    engine.tick(sample,sampled_at=1.08,now=1.08)
    if later is None:sample.pop('ack_sampled_at')
    else:sample['ack_sampled_at']=later
    with pytest.raises(RuntimeError,match='receipt'):
        engine.tick(sample,sampled_at=1.1,now=1.1)
    assert gate.stopped and len(gate.calls)==2


def test_physical_producer_time_can_precede_the_raw_worker_receipt():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=1.06)
    # 保留硬件生产端记录的数据年龄，因此该样本映射到略早的时间。
    engine.tick(sample,sampled_at=1.04,now=1.06)
    assert engine.state=='READY' and not gate.stopped


@pytest.mark.parametrize('worker_time',[None,True,float('nan'),float('inf'),1.22,.9,1.01])
def test_invalid_current_worker_receipt_cannot_validate_an_ack(worker_time):
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample.update(ack_sampled_at=1.05,worker_sampled_at=worker_time)
    with pytest.raises(RuntimeError,match='receipt'):
        engine.tick(sample,sampled_at=1.219,now=1.219)
    assert gate.stopped and len(gate.calls)==1


def test_partial_receipt_cannot_select_legacy_deadline_behavior():
    cfg,gate,engine=pending();sample=feedback(cfg,123,state=0,seq=1)
    sample['worker_sampled_at']=1.05
    with pytest.raises(RuntimeError,match='receipt'):
        engine.tick(sample,sampled_at=1.05,now=1.05)
    assert gate.stopped and len(gate.calls)==1
