import time
import math
import pytest
from leica_tracker.reference_gate import ReferenceGate, rotation_vector_distance

POSE=[-.1,-.5,.2,0,3.,-.5]
ZERO=[0.]*6


def test_rotation_vector_distance_handles_equivalent_and_small_rotations():
    assert rotation_vector_distance([math.pi, 0, 0], [-math.pi, 0, 0]) == pytest.approx(0, abs=1e-14)
    assert rotation_vector_distance([0, 0, 0], [1e-12, 0, 0]) == pytest.approx(1e-12, rel=1e-12)
    assert rotation_vector_distance([0, 0, 0], [0, 0, 0]) == 0


def test_rotation_vector_distance_compares_different_rotation_axes():
    expected = 2 * math.acos(math.cos(.4 / 2) * math.cos(.3 / 2))
    assert rotation_vector_distance([.4, 0, 0], [0, .3, 0]) == pytest.approx(expected)


@pytest.mark.parametrize('delta,allowed', [(.005 - 1e-8, True), (.005, True), (.005 + 1e-8, False)])
def test_gate_applies_same_angle_threshold(delta, allowed):
    g = ReferenceGate()
    sid = g.begin([0] * 6, 10.)
    g.acknowledge()
    for i in range(8):
        t = 10 + i * .1
        g.sample(sid, [0, 0, 0, delta, 0, 0], ZERO, t, t)
    if allowed:
        g.release(sid, 10.71)
        assert not g.blocked
    else:
        with pytest.raises(ValueError):
            g.release(sid, 10.71)
        assert g.blocked


def test_gate_treats_opposite_pi_representations_as_same_pose():
    g = ReferenceGate()
    sid = g.begin([0, 0, 0, math.pi, 0, 0], 10.)
    g.acknowledge()
    for i in range(8):
        t = 10 + i * .1
        g.sample(sid, [0, 0, 0, -math.pi, 0, 0], ZERO, t, t)
    g.release(sid, 10.71)
    assert not g.blocked

def test_gate_only_releases_after_fresh_measured_reference_and_dwell():
    g=ReferenceGate();sid=g.begin(POSE,10.)
    assert g.held(10_000_000_001)
    g.acknowledge()
    with pytest.raises(ValueError):g.release(sid,10.1)
    for i in range(8):g.sample(sid,POSE,ZERO,10.+i*.1,10.+i*.1)
    g.release(sid,10.71)
    assert g.held(10_700_000_000)  # 不复用放行前采集的画面。
    assert not g.held(10_720_000_000)

def test_interruption_or_bad_measurements_do_not_unlock_gate():
    g=ReferenceGate();sid=g.begin(POSE,10.)
    g.acknowledge()
    for i in range(4):g.sample(sid,POSE,ZERO,10.+i*.1,10.+i*.1)
    moved=POSE.copy();moved[0]+=.005
    g.sample(sid,moved,ZERO,10.4,10.4)
    for i in range(5,9):g.sample(sid,POSE,ZERO,10.+i*.1,10.+i*.1)
    with pytest.raises(ValueError):g.release(sid,10.81)
    with pytest.raises(ValueError):g.sample('wrong',POSE,ZERO,10.9,10.9)
    with pytest.raises(ValueError):g.sample(sid,POSE,ZERO,10.,11.)
    assert g.held(11_000_000_000)

def test_duplicate_or_gapped_samples_cannot_manufacture_a_settle_window():
    g=ReferenceGate();sid=g.begin(POSE,10.)
    g.acknowledge()
    g.sample(sid,POSE,ZERO,10.1,10.1)
    with pytest.raises(ValueError):g.sample(sid,POSE,ZERO,10.1,10.1)
    g.sample(sid,POSE,ZERO,10.7,10.7)
    with pytest.raises(ValueError):g.release(sid,10.71)
    assert g.held(10_800_000_000)

@pytest.mark.parametrize('bad',[[0]*5,[float('nan')]*6, 'bad'])
def test_invalid_reference_never_arms(bad):
    with pytest.raises(ValueError):ReferenceGate().begin(bad,10.)


def test_http_gate_is_paired_loopback_and_fails_closed():
    import json,urllib.request,urllib.error
    from leica_tracker.guest_display import GuestDisplay
    with GuestDisplay(port=0) as d:
        h={'X-Guest-Token':d.token,'X-Guest-Client':'physical-supervisor','Origin':d.origin,'Content-Type':'application/json'}
        def post(value,headers=h):
            return urllib.request.urlopen(urllib.request.Request(d.origin+'/reference-gate',data=json.dumps(value).encode(),headers=headers),timeout=2)
        with pytest.raises(urllib.error.HTTPError):post({'op':'begin','reference':POSE},{**h,'X-Guest-Token':'wrong'})
        with post({'op':'begin','reference':POSE}) as r:sid=json.load(r)['session']
        with pytest.raises(urllib.error.HTTPError):post({'op':'release','session':sid})
        with urllib.request.urlopen(urllib.request.Request(d.origin+'/operator-state',headers=h)) as r:
            gate=json.load(r)['reference_gate'];assert gate['blocked'] and not gate['acknowledged']
        from leica_tracker.person_guidance import GuidanceCue,GuidanceDecision
        d.publish(None,GuidanceDecision('PHYSICAL_REFERENCE_HOLD',GuidanceCue.HOLD,(.5,.3,.2),None,None,False,'test'))
        with urllib.request.urlopen(urllib.request.Request(d.origin+'/operator-state',headers=h)) as r:
            assert json.load(r)['reference_gate']['acknowledged']
