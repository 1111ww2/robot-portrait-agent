import json
import time
import urllib.request
import urllib.error

import cv2
import numpy as np
import pytest


def jpeg():
    return cv2.imencode('.jpg',np.full((72,128,3),100,np.uint8))[1].tobytes()


def test_latest_frame_is_single_use_and_uses_challenge_time():
    from leica_tracker.ipad_camera import IPadCapture
    clock=[100.];capture=IPadCapture(clock=lambda:clock[0],read_timeout=0)
    session=capture.begin('ipad')
    ticket=capture.ticket('ipad',session)
    clock[0]=100.1;capture.accept('ipad',session,ticket,jpeg())
    ok,frame=capture.read()
    assert ok and frame.shape==(72,128,3)
    assert capture.source_monotonic_ns==100_000_000_000
    assert capture.read()==(False,None)
    with pytest.raises(ValueError):capture.accept('ipad',session,ticket,jpeg())
    ticket=capture.ticket('ipad',session)
    clock[0]=100.41
    with pytest.raises(ValueError):capture.accept('ipad',session,ticket,jpeg())


def test_camera_owner_cancellation_and_old_buffer_never_replay():
    from leica_tracker.ipad_camera import IPadCapture
    clock=[10.];c=IPadCapture(clock=lambda:clock[0],read_timeout=0)
    session=c.begin('one');ticket=c.ticket('one',session)
    with pytest.raises(ValueError):c.begin('two')
    with pytest.raises(ValueError):c.accept('two',session,ticket,jpeg())
    c.accept('one',session,ticket,jpeg())
    clock[0]+=.31;assert c.read()==(False,None)
    ticket=c.ticket('one',session);c.end('one',session)
    with pytest.raises(ValueError):c.accept('one',session,ticket,jpeg())
    session2=c.begin('two');assert session2!=session


@pytest.mark.parametrize('data',[b'not-jpeg',b'x'*(524288+1),b'\xff\xd8\xff\xc0\x00\x11\x08\x7f\xff\x7f\xff'+b'\x00'*12],ids=['malformed','too-many-bytes','too-many-pixels'])
def test_rejects_malformed_oversize_images_before_decode(data):
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture(read_timeout=0);s=c.begin('ipad');t=c.ticket('ipad',s)
    with pytest.raises(ValueError):c.accept('ipad',s,t,data)
    assert c.read()==(False,None)


def test_paired_guest_camera_upload_and_stop():
    from leica_tracker.ipad_camera import IPadCapture
    from leica_tracker.guest_display import GuestDisplay
    c=IPadCapture(read_timeout=0)
    with GuestDisplay(port=0,remote_capture=c) as d:
        headers={'X-Guest-Token':d.token,'X-Guest-Client':'ipad','Origin':d.origin}
        def post(path,body,extra=None):
            r=urllib.request.Request(d.origin+path,data=body,headers={**headers,'Content-Type':'application/json',**(extra or {})})
            return urllib.request.urlopen(r,timeout=2)
        with post('/camera/start',b'{}') as r:s=json.load(r)['session']
        with post('/camera/ticket',json.dumps({'session':s}).encode()) as r:t=json.load(r)['ticket']
        upload={'Content-Type':'image/jpeg','X-Camera-Session':s,'X-Camera-Ticket':t}
        with post('/camera/frame',jpeg(),upload) as r:assert r.status==200
        assert c.read()[0]
        with pytest.raises(urllib.error.HTTPError):post('/camera/frame',jpeg(),upload)
        with post('/camera/stop',json.dumps({'session':s}).encode()) as r:assert r.status==200
        with pytest.raises(urllib.error.HTTPError):post('/camera/start',b'{}',{'X-Guest-Token':'wrong'})


def test_ipad_source_waits_without_crashing_or_issuing_model_requests():
    import io
    from types import SimpleNamespace
    from leica_tracker.ipad_camera import IPadCapture
    from leica_tracker.app import AppConfig,process_frames
    from leica_tracker.guest_display import GuestDisplay
    c=IPadCapture(read_timeout=.01)
    with GuestDisplay(port=0,remote_capture=c) as guest:
        out=io.StringIO()
        stats=process_frames(c,SimpleNamespace(),AppConfig(camera_source='ipad',guidance_profile='claude-agent',headless=True,duration=.25,max_read_failures=1),guest_display=guest,output=out)
        assert stats.frames_processed==0 and guest.active is False
        events=[json.loads(line) for line in out.getvalue().splitlines()]
        assert events and all(e['cue']=='HOLD' for e in events)


def test_ipad_cli_requires_guest_display_and_sets_mirror_contract():
    from leica_tracker.app import build_parser,_config_from_args
    parser=build_parser()
    with pytest.raises(ValueError):_config_from_args(parser.parse_args(['--camera-source','ipad']))
    config=_config_from_args(parser.parse_args(['--camera-source','ipad','--guest-display','--headless']))
    assert config.source_mirrored and config.camera_source=='ipad'


def test_transport_connection_survives_short_frame_gap_but_not_stop():
    from leica_tracker.ipad_camera import IPadCapture
    clock=[10.];c=IPadCapture(clock=lambda:clock[0],read_timeout=0)
    s=c.begin('ipad');t=c.ticket('ipad',s);c.accept('ipad',s,t,jpeg())
    clock[0]+=.4
    assert not c.fresh() and c.connected()
    c.invalidate_pending()
    assert c.connected() and not c.fresh()
    clock[0]+=2.2
    assert not c.connected()
    t=c.ticket('ipad',s);c.accept('ipad',s,t,jpeg())
    assert c.connected()
    c.end('ipad',s)
    assert not c.connected()


def test_upload_stream_freshness_survives_frame_invalidation_but_not_timeout_or_stop():
    from leica_tracker.ipad_camera import IPadCapture
    clock=[20.];c=IPadCapture(clock=lambda:clock[0],read_timeout=0)
    s=c.begin('ipad');t=c.ticket('ipad',s);c.accept('ipad',s,t,jpeg())
    assert c.stream_fresh() and c.fresh()
    c.invalidate_pending()
    assert not c.fresh() and c.stream_fresh()
    clock[0]+=.31
    assert not c.stream_fresh()
    t=c.ticket('ipad',s);c.accept('ipad',s,t,jpeg())
    assert c.stream_fresh()
    c.end('ipad',s)
    assert not c.stream_fresh()


def test_pause_state_preserves_last_picture_and_clears_feedback():
    from leica_tracker.guest_display import GuestDisplay
    from leica_tracker.person_guidance import GuidanceDecision,GuidanceCue
    with GuestDisplay(port=0) as guest:
        decision=GuidanceDecision('GUEST_READY',GuidanceCue.HOLD,(.5,.3,.2),None,None,False,'test')
        guest.publish(np.full((72,128,3),123,np.uint8),decision)
        picture=guest._jpeg;sequence=guest._sequence
        guest._last_publish=0
        guest.publish(None,decision)
        assert guest._jpeg==picture and guest._sequence==sequence
        assert guest._state['phase']=='GUEST_READY'


def test_camera_diagnostics_are_bounded_numbers_only_and_do_not_make_frames_fresh():
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture(read_timeout=0);s=c.begin('ipad')
    c.ticket('ipad',s,diagnostics={'preview_fps':30,'track_muted':True,'configured_fps':60,'private_text':'discard','hidden':False})
    assert c.diagnostics()=={'preview_fps':30,'track_muted':True,'configured_fps':60,'hidden':False}
    assert not c.fresh() and not c.connected()
    c.ticket('ipad',s,diagnostics={'preview_fps':float('nan'),'configured_fps':999999})
    assert c.diagnostics()=={}

def test_photo_flow_version_is_allowlisted_bounded_numeric_diagnostic():
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture();s=c.begin('ipad')
    c.ticket('ipad',s,diagnostics={'photo_flow_version':1,'photo_started':True,
                                  'photo_state':1,'photo_owner':True,'photo_latched':True,
                                  'photo_camera_ready':False,'photo_active':True,'private_text':'discard'})
    assert c.diagnostics()=={'photo_flow_version':1,'photo_started':True,'photo_state':1,
                             'photo_owner':True,'photo_latched':True,'photo_camera_ready':False,'photo_active':True}
    c.ticket('ipad',s,diagnostics={'photo_flow_version':9999})
    assert c.diagnostics()=={}


def test_voice_diagnostics_are_numeric_only_without_transcripts():
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture();s=c.begin('ipad')
    c.ticket('ipad',s,diagnostics={'voice_available':True,'voice_error':1,'voice_results':2,'voice_transcript':'private'})
    assert c.diagnostics()=={'voice_available':True,'voice_error':1,'voice_results':2}


def test_voice_event_timeline_is_bounded_deduplicated_and_contains_no_free_text(capsys):
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture();s=c.begin('ipad')
    events=[[1,10,1,0,1],[2,12,7,6,1],[3,13,8,0,1]]
    c.ticket('ipad',s,{'voice_events':events,'transcript':'PRIVATE WORDS'})
    c.ticket('ipad',s,{'voice_events':events})
    captured=capsys.readouterr()
    assert captured.out==''  # stdout是机器人桥JSON协议，永远不是日志通道。
    lines=captured.err.splitlines()
    assert len(lines)==3
    rows=[json.loads(line.removeprefix('VOICE_EVENT ')) for line in lines]
    assert [r['event'] for r in rows]==[1,7,8]
    assert 'PRIVATE WORDS' not in str(rows)
    c.ticket('ipad',s,{'voice_events':[[4,14,'PRIVATE WORDS',0,1],[5,15,999,0,1]]})
    captured=capsys.readouterr()
    assert captured.out==captured.err==''
    assert 'voice_events' not in c.diagnostics()

def test_camera_upload_stage_timings_are_bounded_numeric_diagnostics():
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture(read_timeout=0);s=c.begin('ipad')
    c.ticket('ipad',s,{'ticket_ms':20,'fresh_frame_ms':10,'encode_ms':40,'upload_ms':30,'error_ms':500,'error_stage':4,'text':'private'})
    assert c.diagnostics()=={'ticket_ms':20,'fresh_frame_ms':10,'encode_ms':40,'upload_ms':30,'error_ms':500,'error_stage':4}
    c.ticket('ipad',s,{'ticket_ms':float('nan'),'fresh_frame_ms':-1,'encode_ms':999999})
    assert c.diagnostics()=={}


def test_control_transport_diagnostics_remain_numeric_bounded_and_do_not_refresh_frames():
    from leica_tracker.ipad_camera import IPadCapture
    c=IPadCapture(read_timeout=0);s=c.begin('ipad')
    values={'control_hb_version':1,'control_hb_expiry':4,'control_hb_max_gap_ms':313,
            'control_hb_last_rtt_ms':152,'control_hb_max_rtt_ms':10000,
            'control_hb_pending':3,'control_hb_max_pending':3,'control_hb_sent':240,
            'control_hb_settled':237,'control_hb_aborted':5}
    c.ticket('ipad',s,{**values,'control_hb_token':'private'})
    assert c.diagnostics()==values
    assert not c.stream_fresh()
    c.ticket('ipad',s,{'control_hb_max_gap_ms':10001,'control_hb_last_rtt_ms':float('nan'),
                       'control_hb_sent':241,'control_hb_expiry':'private','control_hb_pending':-1})
    assert c.diagnostics()=={}
    c.release()


def test_camera_first_error_and_maxima_are_allowlisted_bounded_without_refreshing_stream():
    from leica_tracker.ipad_camera import IPadCapture
    capture=IPadCapture();session=capture.begin('ipad')
    values={'camera_first_error_stage':4,'camera_first_error_kind':1,'camera_first_error_status':503,
            'camera_first_error_ms':10000,'camera_max_ticket_ms':120,'camera_max_fresh_frame_ms':180,
            'camera_max_encode_ms':17,'camera_max_upload_ms':29,'camera_max_upload_ack_gap_ms':313,
            'camera_error_count':240}
    capture.ticket('ipad',session,{**values,'camera_error_text':'private'})
    assert capture.diagnostics()==values
    assert not capture.stream_fresh() and not capture.fresh()
    capture.ticket('ipad',session,{'camera_first_error_stage':5,'camera_first_error_kind':7,
        'camera_first_error_status':600,'camera_first_error_ms':10001,'camera_max_ticket_ms':float('nan'),
        'camera_max_fresh_frame_ms':-1,'camera_max_encode_ms':True,'camera_max_upload_ms':'private',
        'camera_max_upload_ack_gap_ms':10001,'camera_error_count':241})
    assert capture.diagnostics()=={}
    capture.ticket('ipad',session,{'camera_first_error_stage':True,'camera_first_error_kind':1.5,
        'camera_first_error_status':403.5,'camera_error_count':True})
    assert capture.diagnostics()=={}
    capture.release()
