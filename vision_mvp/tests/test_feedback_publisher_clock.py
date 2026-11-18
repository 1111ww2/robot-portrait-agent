"""以确定顺序交错调度发布操作，不使用套接字或机器人。"""
import threading

from leica_tracker.physical_runtime import FeedbackPublisher


class OneWatchIteration:
    def __init__(self):
        self.calls=0
        self.closed=False
    def wait(self,timeout):
        self.calls+=1
        return self.closed or self.calls>1
    def set(self):
        self.closed=True


class Gate:
    def __init__(self):self.reason=None
    def stop(self,reason):self.reason=reason


def publisher():
    # 仅构造看门狗读取的字段，不启动发送端。
    p=object.__new__(FeedbackPublisher)
    p.max_latency=.25;p._pending={};p._lock=threading.Lock()
    p.recovery=None  # 验证既有的终止截止时间策略。
    p._closed=OneWatchIteration();p.gate=Gate();p.failed=None
    return p


def test_pending_submission_between_prelock_clock_and_lock_is_not_expired():
    p=publisher();now=[100.]
    class SubmitJustBeforeLock:
        """模拟提交操作恰好在看门狗之前获取锁的情况。"""
        entered=False
        def __enter__(self):
            if not self.entered:
                self.entered=True
                now[0]=100.001
                p._pending[1]=now[0]
        def __exit__(self,*args):return False
    p.clock=lambda:now[0];p._lock=SubmitJustBeforeLock()
    p._watchdog()
    assert p.failed is None
    assert p.gate.reason is None


def test_real_deadline_expiry_still_stops_at_original_250_ms_limit():
    p=publisher();p.clock=lambda:100.;p._pending[1]=99.749
    p._watchdog()
    assert p.failed=='Physical feedback deadline exceeded'
    assert p.gate.reason==p.failed


def test_original_deadline_boundary_remains_valid():
    p=publisher();p.clock=lambda:100.;p._pending[1]=99.75
    p._watchdog()
    assert p.failed is None and p.gate.reason is None
