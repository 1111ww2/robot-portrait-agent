"""显示端收到更新，不能因此取得电机控制租约。"""
import threading
import time
from types import SimpleNamespace

from leica_tracker.physical_runtime import FeedbackPublisher


def test_display_http_failure_does_not_stop_control_heartbeat():
    seen=[];stops=[];display_attempted=threading.Event();heartbeat=threading.Event()
    class Display:
        def feedback_request(self,body):
            display_attempted.set();raise TimeoutError('display delayed')
        def close_feedback_transport(self):pass
    class Reference:
        def presentation_transport(self):return Display()
        def feedback_request(self,body):seen.append(body['op']);heartbeat.set();return {'ok':True}
        def close_feedback_transport(self):pass
    pub=FeedbackPublisher(Reference(),SimpleNamespace(stop=stops.append))
    try:
        pub.submit({'op':'begin','action_id':1})
        assert display_attempted.wait(1)
        pub.submit({'op':'heartbeat'})
        assert heartbeat.wait(1)
        assert seen==['heartbeat'] and stops==[] and pub.failed is None
        assert pub.presentation_failures==1
    finally:pub.close()


def test_blocked_display_sender_does_not_block_control_thread():
    entered=threading.Event();release=threading.Event();heartbeat=threading.Event();stops=[]
    class Display:
        def feedback_request(self,body):entered.set();release.wait(1);return {'ok':True}
        def close_feedback_transport(self):pass
    class Reference:
        def presentation_transport(self):return Display()
        def feedback_request(self,body):heartbeat.set();return {'ok':True}
        def close_feedback_transport(self):pass
    pub=FeedbackPublisher(Reference(),SimpleNamespace(stop=stops.append))
    try:
        pub.submit({'op':'begin','action_id':1});assert entered.wait(1)
        pub.submit({'op':'heartbeat'});assert heartbeat.wait(.2)
        assert stops==[] and pub.failed is None
    finally:release.set();pub.close()


def test_identical_queued_pulses_coalesce_but_inflight_deadline_is_preserved():
    entered=threading.Event();release=threading.Event();seen=[];stops=[]
    class Reference:
        def feedback_request(self,body):
            seen.append(dict(body));entered.set();release.wait(.2);return {'ok':True}
        def close_feedback_transport(self):pass
    pub=FeedbackPublisher(Reference(),SimpleNamespace(stop=stops.append))
    try:
        pub.submit({'op':'heartbeat'});assert entered.wait(1)
        original=min(pub._pending.values())
        for _ in range(8):pub.submit({'op':'heartbeat'})
        assert pub.queue.qsize()==1
        assert len(pub._pending)==2 and min(pub._pending.values())==original
        # 状态转换和之前的脉冲是不同事件，不能互相替代。
        pub.submit({'op':'intro','intro_id':1,'status':'active'})
        pub.submit({'op':'intro','intro_id':1,'status':'ready'})
        assert pub.queue.qsize()==3
    finally:release.set();pub.close()
